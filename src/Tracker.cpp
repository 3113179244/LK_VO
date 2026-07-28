#include "Tracker.h"
#include <iostream>

// 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
{
    mpFeatureDetector = std::make_shared<FeatureDetector>();
    mpClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    std::cout << "[Tracker] Initialized successfully." << std::endl;
}

Eigen::Matrix4d Tracker::GrabImageStereo(const double timestamp, const cv::Mat &image0, const cv::Mat &image1, cv::Mat &matDisplay)
{
    cv::Mat image0Clahe, image1Clahe;
    if (!image0.empty())
        mpClahe->apply(image0, image0Clahe);
    if (!image1.empty())
        mpClahe->apply(image1, image1Clahe);

    mpCurrFrame = std::make_shared<Frame>(timestamp, image0Clahe, image1Clahe, mNextFrameId++);

    // 提取与追踪特征点
    if (mpFeatureDetector)
    {
        mpFeatureDetector->TrackImage(mpPrevFrame, mpCurrFrame, mPrevImage0, image0Clahe, image1Clahe);
        mpFeatureDetector->DrawFeaturesOnImage(
            image0,
            image1,
            mpCurrFrame->mvleftpixel,
            mpCurrFrame->mvrightpixel,
            mpCurrFrame->mvTrackCnt,
            matDisplay);
    }

    mpPrevFrame = mpCurrFrame;
    mPrevImage0 = image0Clahe.clone();
    return Eigen::Matrix4d::Identity();
}