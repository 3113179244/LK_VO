#include "Tracker.h"
#include <iostream>

// 1. 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
{
    std::cout << "[Tracker] Initialized successfully." << std::endl;
}

// 2. 实现 GrabImageStereo
Eigen::Matrix4d Tracker::GrabImageStereo(const double timestamp, const cv::Mat& image0, const cv::Mat& image1)
{
    // 临时返回单位阵，后续写入真正的跟踪算法
    return Eigen::Matrix4d::Identity();
}