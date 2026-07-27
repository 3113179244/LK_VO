#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>

#include "Config.h"
#include "Camera.h"
#include "Frame.h"
#include "FeatureDetector.h"
#include "Map.h"
#include "Optimizer.h"

class Tracker {
public:
    Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap);
    ~Tracker() = default;

    // 系统的主入口：传入当前时间戳和左右目图像，返回当前位姿 Tcw
    Eigen::Matrix4d GrabImageStereo(const double timestamp, const cv::Mat& image0, const cv::Mat& image1);

private:

};

#endif // TRACKER_H