#include "Tracker.h"
#include <iostream>

// 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
{
    std::cout << "[Tracker] Initialized successfully." << std::endl;
}

Eigen::Matrix4d Tracker::GrabImageStereo(const double timestamp, const cv::Mat& image0, const cv::Mat& image1)
{
    return Eigen::Matrix4d::Identity();
}