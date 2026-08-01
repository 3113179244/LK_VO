#include "FrameDrawer.h"
#include "Tracker.h"
#include "Map.h"
#include "MapPoint.h"

FrameDrawer::FrameDrawer(Map* pMap) : mpMap(pMap), mbOnlyTracking(false), mnTracked(0), mnTrackedVO(0)
{
    mState = Tracker::SYSTEM_NOT_READY;
    mIm = cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    std::vector<cv::KeyPoint> vIniKeys;
    std::vector<int> vMatches;
    std::vector<cv::KeyPoint> vCurrentKeys;
    std::vector<bool> vbVO, vbMap;
    int state;

    // 加锁拷贝线程安全数据
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

    // 绘制逻辑
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
        const float r = 5;
        const int n = vCurrentKeys.size();
        for (int i = 0; i < n; i++)
        {
            if (vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1(vCurrentKeys[i].pt.x - r, vCurrentKeys[i].pt.y - r);
                cv::Point2f pt2(vCurrentKeys[i].pt.x + r, vCurrentKeys[i].pt.y + r);

                // 匹配到 MapPoint (全局地图点)：绘制绿色方框加实心圆
                if (vbMap[i])
                {
                    cv::rectangle(im, pt1, pt2, cv::Scalar(0, 255, 0));
                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 255, 0), -1);
                    mnTracked++;
                }
                // 匹配到 VO 临时点：绘制蓝色方框加实心圆
                else
                {
                    cv::rectangle(im, pt1, pt2, cv::Scalar(255, 0, 0));
                    cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(255, 0, 0), -1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);
    return imWithInfo;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    std::stringstream s;
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

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    im.copyTo(imText.rowRange(0, im.rows).colRange(0, im.cols));
    imText.rowRange(im.rows, imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, 8);
}

void FrameDrawer::Update(Tracker *pTracker)
{
    std::unique_lock<std::mutex> lock(mMutex);
    
    // 拷贝图像与关键点
    pTracker->mCurrentFrame.mImLeft.copyTo(mIm);
    mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = std::vector<bool>(N, false);
    mvbMap = std::vector<bool>(N, false);

    if (pTracker->mState == Tracker::OK)
    {
        for (int i = 0; i < N; i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if (pMP)
            {
                if (!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if (pMP->GetObservations().size() > 0)
                        mvbMap[i] = true;
                    else
                        mvbVO[i] = true;
                }
            }
        }
    }
    mState = static_cast<int>(pTracker->mState);
}