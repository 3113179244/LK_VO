#include "Tracker.h"
#include <iostream>

// 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
{
    mpFeatureDetector = std::make_shared<FeatureDetector>();
    std::cout << "[Tracker] Initialized successfully." << std::endl;
}

Eigen::Matrix4d Tracker::GrabImageStereo(const double timestamp, const cv::Mat &image0, const cv::Mat &image1, cv::Mat &matDisplay)
{
    mpCurrFrame = std::make_shared<Frame>(timestamp, image0, image1, mNextFrameId++);

    // 提取与追踪特征点
    if (mpFeatureDetector)
    {
        mpFeatureDetector->TrackImage(mpPrevFrame, mpCurrFrame, mPrevImage0, image0, image1);
        mpFeatureDetector->DrawFeaturesOnImage(
            image0,
            image1,
            mpCurrFrame->mvleftpixel,
            mpCurrFrame->mvrightpixel,
            mpCurrFrame->mvTrackCnt,
            matDisplay);
        std::cout << "Frame " << mpCurrFrame->mFrameId << " features: " << mpCurrFrame->mvleftpixel.size() << std::endl;
    }

    mpPrevFrame = mpCurrFrame;
    mPrevImage0 = image0.clone();
    return Eigen::Matrix4d::Identity();
}