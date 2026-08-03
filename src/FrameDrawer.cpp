#include "FrameDrawer.h"
#include "Tracker.h"
#include "Map.h"
#include "MapPoint.h"

// 构造函数：初始化相关参数与默认画布（默认分配黑底图像）
FrameDrawer::FrameDrawer(Map* pMap) : mpMap(pMap), mbOnlyTracking(false), mnTracked(0), mnTrackedVO(0)
{
    mState = Tracker::SYSTEM_NOT_READY;
    mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
}

/**
 * @brief 绘制并生成用于渲染展示的图像帧
 */
cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    std::vector<cv::KeyPoint> vIniKeys;
    std::vector<int> vMatches;
    std::vector<cv::KeyPoint> vCurrentKeys;
    std::vector<bool> vbVO, vbMap;
    int state;

    // 加锁拷贝数据：防止在绘制过程中 Tracker 线程修改数据造成数据竞态
    {
        std::unique_lock<std::mutex> lock(mMutex);
        state = mState;
        if (mState == Tracker::SYSTEM_NOT_READY)
            mState = Tracker::NO_IMAGES_YET;

        mIm.copyTo(im); // 深拷贝当前帧图像

        // 根据不同状态，提取相应的局部特征和匹配数据
        if (mState == Tracker::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if (mState == Tracker::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if (mState == Tracker::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    } // 自动解锁

    // 若原图为单通道灰度图，需转换为三通道 BGR 彩色图以便绘制彩色线和方框
    if (im.channels() < 3)
    {
        cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }

    // 图像绘制逻辑
    
    // 状态 A: 未初始化阶段（单目初始化过程，绘制两帧特征点间的连线/光流轨迹）
    if (state == Tracker::NOT_INITIALIZED)
    {
        for (unsigned int i = 0; i < vMatches.size(); i++)
        {
            if (vMatches[i] >= 0) // 有有效匹配
            {
                // 连接初始化初始帧点与当前帧点（绿色线）
                cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt, cv::Scalar(0, 255, 0));
            }
        }
    }
    // 状态 B: 跟踪正常阶段
    else if (state == Tracker::OK)
    {
        mnTracked = 0;
        mnTrackedVO = 0;
        const float r = 5; // 标记方框的半径（边长的一半）
        const int n = vCurrentKeys.size();
        
        for (int i = 0; i < n; i++)
        {
            if (vbVO[i] || vbMap[i])
            {
                // 计算特征点四周方框的左上角与右下角坐标
                cv::Point2f pt1(vCurrentKeys[i].pt.x - r, vCurrentKeys[i].pt.y - r);
                cv::Point2f pt2(vCurrentKeys[i].pt.x + r, vCurrentKeys[i].pt.y + r);

                // 匹配到 MapPoint (全局地图点)：绘制绿色矩形框 + 绿色实心圆
                if (vbMap[i])
                {
                    cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0));
                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                    mnTracked++;
                }
                // 匹配到 VO 临时点 (通常是没有被共视或新插入的点)：绘制蓝色矩形框 + 蓝色实心圆
                else
                {
                    cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 0));
                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                    mnTrackedVO++;
                }
            }
        }
    }

    // 3. 绘制底部的文字状态栏
    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);
    return imWithInfo;
}

/**
 * @brief 在图像底部追加状态文本区域（如 KF 数量、MP 数量、匹配数等）
 */
void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    std::stringstream s;
    
    // 拼接状态描述字符串
    if (nState == Tracker::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if (nState == Tracker::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if (nState == Tracker::OK)
    {
        s << "SLAM MODE | ";
        int nKFs = mpMap ? mpMap->GetKeyFramesInMap() : 0;
        int nMPs = mpMap ? mpMap->GetMapPointsInMap() : 0;
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if (mnTrackedVO > 0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if (nState == Tracker::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if (nState == Tracker::SYSTEM_NOT_READY)
    {
        s << " LOADING SYSTEM. PLEASE WAIT...";
    }

    // 计算文字所需的像素高宽
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    // 创建新图像，高度增加 (textSize.height + 10) 像素以容纳文字底色栏
    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    
    // 拷贝原图像到上半部分
    im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
    
    // 将下半部分背景填充为黑色 (全 0)
    imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    
    // 在底部黑色背景区域绘制白色文本
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

/**
 * @brief 供主线程/Tracker 调用的更新接口，用于同步当前帧的特征数据和匹配标记
 */
void FrameDrawer::Update(Tracker *pTracker)
{
    std::unique_lock<std::mutex> lock(mMutex);
    
    // 1. 拷贝当前帧图像和提取的 KeyPoints 关键点
    pTracker->mCurrentFrame.mImGrayLeft.copyTo(mIm);
    mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    
    // 重置标记数组
    mvbVO = std::vector<bool>(N, false);
    mvbMap = std::vector<bool>(N, false);

    // 2. 如果跟踪正常，统计和归类每个特征点对应的地图点类型
    if (pTracker->mState == Tracker::OK)
    {
        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                // 过滤掉当前帧中的离群点/外点 (Outliers)
                if (!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    // 观测数 > 0 表示属于建立好的全局 MapPoint（建图点）
                    if (pMP->GetObservations().size() > 0)
                        mvbMap[i] = true;
                    // 否则视为临时 VO 点（里程计点）
                    else
                        mvbVO[i] = true;
                }
            }
        }
    }
    
    // 3. 更新系统状态
    mState = static_cast<int>(pTracker->mState);
}