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

    // 加锁拷贝数据
    {
        std::unique_lock<std::mutex> lock(mMutex);
        state = mState;
        if (mState == Tracker::SYSTEM_NOT_READY)
            mState = Tracker::NO_IMAGES_YET;

        mIm.copyTo(im);

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
    }

    if (im.channels() < 3)
    {
        cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }

    // 图像绘制逻辑
    if (state == Tracker::NOT_INITIALIZED)
    {
        for (unsigned int i = 0; i < vMatches.size(); i++)
        {
            if (vMatches[i] >= 0)
            {
                cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt, cv::Scalar(0, 255, 0));
            }
        }
    }
    else if (state == Tracker::OK)
    {
        mnTracked = 0;
        mnTrackedVO = 0;
        const float r = 5; // 矩形框半边长
        const int n = vCurrentKeys.size();
        
        for (int i = 0; i < n; i++)
        {
            // 只要匹配到了地图点，或者有提取到特征点
            if (vbMap[i] || vbVO[i])
            {
                cv::Point2f pt1(vCurrentKeys[i].pt.x - r, vCurrentKeys[i].pt.y - r);
                cv::Point2f pt2(vCurrentKeys[i].pt.x + r, vCurrentKeys[i].pt.y + r);

                // 绘制外层绿色矩形框 + 内部实心圆点
                cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0), 1);
                cv::circle(im, vCurrentKeys[i].pt, 1, cv::Scalar(0, 255, 0), -1);
                mnTracked++;
            }
        }

        // 如果目前 Tracker 还没跟踪上（Matches 为 0），默认将提取出的所有特征点都画上“方框+圆点”
        if (mnTracked == 0) 
        {
            for (int i = 0; i < n; i++) 
            {
                cv::Point2f pt1(vCurrentKeys[i].pt.x - r, vCurrentKeys[i].pt.y - r);
                cv::Point2f pt2(vCurrentKeys[i].pt.x + r, vCurrentKeys[i].pt.y + r);

                cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0), 1);
                cv::circle(im, vCurrentKeys[i].pt, 1, cv::Scalar(0, 255, 0), -1);
            }
        }
    }

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
    
    // 拷贝当前图像和特征点
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    
    // 重置标记数组
    mvbVO = std::vector<bool>(N, false);
    mvbMap = std::vector<bool>(N, false);

    // 统计和归类每个特征点对应的地图点类型
    if (pTracker->mState == Tracker::OK)
    {
        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    // 【修改点】：只要当前帧绑定了 MapPoint 并且不是 Outlier，就标记为地图点画框
                    mvbMap[i] = true;
                }
            }
        }
    }
    
    mState = static_cast<int>(pTracker->mState);
}

