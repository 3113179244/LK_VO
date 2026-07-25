#include "FeatureDetector.h"
#include "Config.h" 

int FeatureTracker::nextFeatureId = 0;

FeatureTracker::FeatureTracker()
{
    maxFeatures = Config::get<int>("max_cnt");
    minFeatureDist = Config::get<int>("min_dist");
}

void FeatureTracker::TrackImage(const double &timestamp, const cv::Mat &_currImgLeft, const cv::Mat &_currImgRight)
{
    currImgLeft = _currImgLeft;
    currImgRight = _currImgRight;
    currPtsLeft.clear();

    if (prevPtsLeft.size() > 0)
    {
        TrackTemporal();
    }

    SetMask();
    DetectNewFeatures();

    if (!currImgRight.empty())
    {
        TrackStereo();
    }

    RejectOutliers();

    // 为下一帧更新数据
    prevImgLeft = currImgLeft;
    prevPtsLeft = currPtsLeft;
}

std::vector<TrackedFeature> FeatureTracker::GetCurrentTrackedFeatures() const
{
    std::vector<TrackedFeature> features;
    for (size_t i = 0; i < currPtsLeft.size(); i++)
    {
        TrackedFeature f;
        f.id = trackIds[i];
        f.ptLeft = currPtsLeft[i];
        if (i < currPtsRight.size())
            f.ptRight = currPtsRight[i];
        f.trackCount = trackCnt[i];
        features.push_back(f);
    }
    return features;
}

void FeatureTracker::TrackTemporal()
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImgLeft, currImgLeft, prevPtsLeft, currPtsLeft, status, err, cv::Size(21, 21), 3);

    // 筛选出追踪成功的点
    std::vector<cv::Point2f> goodPts;
    std::vector<int> goodIds;
    std::vector<int> goodCnt;
    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            goodPts.push_back(currPtsLeft[i]);
            goodIds.push_back(trackIds[i]);
            goodCnt.push_back(trackCnt[i] + 1);
        }
    }
    currPtsLeft = goodPts;
    trackIds = goodIds;
    trackCnt = goodCnt;
}

void FeatureTracker::TrackStereo()
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(currImgLeft, currImgRight, currPtsLeft, currPtsRight, status, err, cv::Size(21, 21), 3);
    // 这里可进一步处理左右眼由于极线约束未对齐的异常点
}

void FeatureTracker::RejectOutliers()
{
    if (currPtsLeft.size() >= 8 && currPtsRight.size() >= 8)
    {
        std::vector<uchar> status;
        cv::findFundamentalMat(currPtsLeft, currPtsRight, cv::FM_RANSAC, 3.0, 0.99, status);
        // 根据 status 去除外点 (这里简化)
    }
}

void FeatureTracker::DetectNewFeatures()
{
    int maxToDetect = maxFeatures - currPtsLeft.size();
    if (maxToDetect > 0)
    {
        std::vector<cv::Point2f> newPts;
        cv::goodFeaturesToTrack(currImgLeft, newPts, maxToDetect, 0.01, minFeatureDist, mask);

        for (auto &pt : newPts)
        {
            currPtsLeft.push_back(pt);
            trackIds.push_back(nextFeatureId++);
            trackCnt.push_back(1);
        }
    }
}

void FeatureTracker::SetMask()
{
    mask = cv::Mat(currImgLeft.size(), CV_8UC1, cv::Scalar(255));
    for (const auto &pt : currPtsLeft)
    {
        cv::circle(mask, pt, minFeatureDist, 0, -1);
    }
}