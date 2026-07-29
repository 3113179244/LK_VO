#include "Tracker.h"
#include "Camera.h"     
#include "Map.h"        
#include "MapPoint.h"
#include <iostream>

// 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
    : mpCamera(pCamera), mpMap(pMap),
      mbInitialized(false), mNextMapPointId(0)
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
    if (!mbInitialized && mpCurrFrame->mvleftpixel.size() > 0)
    {
        const int N = static_cast<int>(mpCurrFrame->mvleftpixel.size());
        // 准备相机内参和基线
        double fx = mpCamera->fx;
        double fy = mpCamera->fy;
        double cx = mpCamera->cx;
        double cy = mpCamera->cy;
        double baseline = mpCamera->mBaseline;

        // 确保地图点容器大小匹配
        mpCurrFrame->mvpMapPoints.resize(N, nullptr);

        // 遍历特征点
        for (int i = 0; i < N; ++i)
        {
            const cv::Point2f &ptLeft = mpCurrFrame->mvleftpixel[i].pt;
            const cv::KeyPoint &kpRight = mpCurrFrame->mvrightpixel[i];
            if (kpRight.pt.x < 0) // 无效的双目匹配
                continue;

            float disp = ptLeft.x - kpRight.pt.x;
            if (disp <= 0.0f)
                continue;

            float depth = static_cast<float>(fx * baseline / disp);
            if (depth < 0.1f || depth > 50.0f) // 深度范围过滤
                continue;

            // 归一化坐标 -> 世界坐标（初始位姿为单位矩阵）
            float u = (ptLeft.x - static_cast<float>(cx)) / static_cast<float>(fx);
            float v = (ptLeft.y - static_cast<float>(cy)) / static_cast<float>(fy);
            Eigen::Vector3d P_world = depth * Eigen::Vector3d(u, v, 1.0);

            // 构建 MapPoint
            cv::Mat pos(3, 1, CV_32F);
            pos.at<float>(0) = static_cast<float>(P_world.x());
            pos.at<float>(1) = static_cast<float>(P_world.y());
            pos.at<float>(2) = static_cast<float>(P_world.z());

            auto pMP = std::make_shared<MapPoint>(mNextMapPointId++, pos);
            pMP->AddObservation(mpCurrFrame, i); // 添加观测
            mpMap->InsertMapPoint(pMP);          // 加入地图
            mpCurrFrame->mvpMapPoints[i] = pMP;  // 关联至当前帧
        }

        // 将当前帧作为关键帧插入地图（供可视化用）
        mpMap->InsertKeyFrame(mpCurrFrame);
        mbInitialized = true;
        std::cout << "[Tracker] Initial map generated with "
                  << mpCurrFrame->mvpMapPoints.size() << " points." << std::endl;
    }
    mpPrevFrame = mpCurrFrame;
    mPrevImage0 = image0.clone();
    return Eigen::Matrix4d::Identity();
}