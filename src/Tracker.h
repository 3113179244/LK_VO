#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <memory>
#include <mutex>

// 前向声明
class Camera;
class Map;
class Frame;
class FeatureDetector;
class Optimizer;

class Tracker {
public:
    Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap);
    ~Tracker() = default;

    Eigen::Matrix4d GrabImageStereo(const double timestamp, const cv::Mat& image0, const cv::Mat& image1);

private:
    std::shared_ptr<Camera> mpCamera;
    std::shared_ptr<Map> mpMap;
};

#endif // TRACKER_H