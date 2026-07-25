#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureDetector {
public:
    FeatureDetector();

    void TrackImage(const double &timestamp, const cv::Mat &_currImgLeft, const cv::Mat &_currImgRight);

    // Getter 声明
    const std::vector<int>& getTrackIds() const;
    const std::vector<cv::Point2f>& getCurrPtsLeft() const;
    const std::vector<cv::Point2f>& getCurrPtsRight() const;
    const std::vector<int>& getTrackCnt() const;

private:
    void TrackTemporal();
    void TrackStereo();
    void RejectOutliers();
    void DetectNewFeatures();
    void SetMask();
    void PrioritizeOldFeatures();// 保老点
    bool inBorder(const cv::Point2f &pt, int cols, int rows);
private:
    // 图像缓存
    cv::Mat prevImgLeft;
    cv::Mat currImgLeft;
    cv::Mat currImgRight;

    // 2. 清理了重复定义的成员变量，只留一份
    std::vector<int> trackIds;             // 特征点 ID
    std::vector<cv::Point2f> prevPtsLeft;  // 上一帧左目坐标
    std::vector<cv::Point2f> currPtsLeft;  // 当前帧左目坐标
    std::vector<cv::Point2f> currPtsRight; // 当前帧右目坐标
    std::vector<int> trackCnt;             // 连续追踪次数

    cv::Mat mask;

    int maxFeatures;
    int minFeatureDist;
    static int nextFeatureId;
};

#endif // FEATURE_DETECTOR_H