#include "FeatureDetector.h"
#include "Config.h" 

int FeatureDetector::nextFeatureId = 0;

FeatureDetector::FeatureDetector()
{
    maxFeatures = Config::get<int>("max_cnt");
    minFeatureDist = Config::get<int>("min_dist");
}

void FeatureDetector::TrackImage(const double &timestamp, const cv::Mat &_currImgLeft, const cv::Mat &_currImgRight)
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

// （注：原 GetCurrentFeatureDetectors 函数已删除，直接使用下面的 Getter 取数据）

void FeatureDetector::TrackTemporal()
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImgLeft, currImgLeft, prevPtsLeft, currPtsLeft, status, err, cv::Size(21, 21), 3);

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

void FeatureDetector::TrackStereo()
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(currImgLeft, currImgRight, currPtsLeft, currPtsRight, status, err, cv::Size(21, 21), 3);
}

void FeatureDetector::RejectOutliers()
{
    if (currPtsLeft.size() >= 8 && currPtsRight.size() >= 8)
    {
        std::vector<uchar> status;
        cv::findFundamentalMat(currPtsLeft, currPtsRight, cv::FM_RANSAC, 3.0, 0.99, status);
    }
}

void FeatureDetector::DetectNewFeatures()
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

void FeatureDetector::SetMask()
{
    mask = cv::Mat(currImgLeft.size(), CV_8UC1, cv::Scalar(255));
    for (const auto &pt : currPtsLeft)
    {
        cv::circle(mask, pt, minFeatureDist, 0, -1);
    }
}

// Getter 函数实现
const std::vector<int>& FeatureDetector::getTrackIds() const 
{ 
    return trackIds; 
}

const std::vector<cv::Point2f>& FeatureDetector::getCurrPtsLeft() const 
{ 
    return currPtsLeft; 
}

const std::vector<cv::Point2f>& FeatureDetector::getCurrPtsRight() const 
{ 
    return currPtsRight; 
}

const std::vector<int>& FeatureDetector::getTrackCnt() const 
{ 
    return trackCnt; 
}