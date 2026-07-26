#include "FeatureDetector.h"
#include "Config.h"

int FeatureDetector::nextFeatureId = 0;

FeatureDetector::FeatureDetector()
{
    maxFeatures = Config::get<int>("max_cnt");
    minFeatureDist = Config::get<int>("min_dist");
}

void FeatureDetector::TrackImage(const double &timestamp, const cv::Mat &currImgLeft_, const cv::Mat &currImgRight_)
{
    currImgLeft = currImgLeft_;
    currImgRight = currImgRight_;
    currPtsLeft.clear();

    if (prevPtsLeft.size() > 0)
    {
        TrackPrevLeftToCurrLeft();
    }

    SortPointsByTrackCount();

    SetMask();
    DetectNewFeatures();

    if (!currImgRight.empty())
    {
        TrackStereo();
    }

    FilterStereoMismatch();

    prevImgLeft = currImgLeft;
    prevPtsLeft = currPtsLeft;
}

void FeatureDetector::SortPointsByTrackCount()
{
    if (currPtsLeft.empty())
        return;

    std::vector<size_t> indices(currPtsLeft.size());

    for (size_t i = 0; i < indices.size(); ++i)
    {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b)
              { return trackCnt[a] > trackCnt[b]; });

    std::vector<cv::Point2f> sortedPts;
    std::vector<int> sortedIds;
    std::vector<int> sortedCnt;

    sortedPts.reserve(currPtsLeft.size());
    sortedIds.reserve(trackIds.size());
    sortedCnt.reserve(trackCnt.size());

    for (size_t idx : indices)
    {
        sortedPts.push_back(currPtsLeft[idx]);
        sortedIds.push_back(trackIds[idx]);
        sortedCnt.push_back(trackCnt[idx]);
    }

    currPtsLeft = std::move(sortedPts);
    trackIds = std::move(sortedIds);
    trackCnt = std::move(sortedCnt);
}

void FeatureDetector::TrackPrevLeftToCurrLeft()
{
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prevImgLeft, currImgLeft, prevPtsLeft, currPtsLeft, status, err, cv::Size(21, 21), 3);

    std::vector<cv::Point2f> goodPts;
    std::vector<int> goodIds;
    std::vector<int> goodCnt;

    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i] && inBorder(currPtsLeft[i], currImgLeft.cols, currImgLeft.rows))
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

void FeatureDetector::FilterStereoMismatch()
{
    if (currPtsLeft.size() >= 8 && currPtsRight.size() >= 8)
    {
        std::vector<uchar> status;
        cv::findFundamentalMat(currPtsLeft, currPtsRight, cv::FM_RANSAC, 3.0, 0.99, status);

        std::vector<cv::Point2f> goodPtsLeft, goodPtsRight;
        std::vector<int> goodIds, goodCnt;

        for (size_t i = 0; i < status.size(); i++)
        {
            if (status[i])
            {
                goodPtsLeft.push_back(currPtsLeft[i]);
                if (i < currPtsRight.size())
                    goodPtsRight.push_back(currPtsRight[i]);
                goodIds.push_back(trackIds[i]);
                goodCnt.push_back(trackCnt[i]);
            }
        }
        currPtsLeft = goodPtsLeft;
        currPtsRight = goodPtsRight;
        trackIds = goodIds;
        trackCnt = goodCnt;
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
        if (mask.at<uchar>(pt) == 255)
        {
            cv::circle(mask, pt, minFeatureDist, 0, -1);
        }
    }
}

const std::vector<int> &FeatureDetector::getTrackIds() const
{
    return trackIds;
}

const std::vector<cv::Point2f> &FeatureDetector::getCurrPtsLeft() const
{
    return currPtsLeft;
}

const std::vector<cv::Point2f> &FeatureDetector::getCurrPtsRight() const
{
    return currPtsRight;
}

const std::vector<int> &FeatureDetector::getTrackCnt() const
{
    return trackCnt;
}

bool FeatureDetector::inBorder(const cv::Point2f &pt, int cols, int rows)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);

    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}