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
        mpPrevFrame = mpCurrFrame;
        mPrevImage0 = image0.clone();
        std::cout << "Frame " << mpCurrFrame->mFrameId << " features: " << mpCurrFrame->mvleftpixel.size() << std::endl;
    }
    if (!mbInitialized && mpCurrFrame->mvleftpixel.size() > 0)
    {
        const int N = static_cast<int>(mpCurrFrame->mvleftpixel.size());
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
            const cv::Point2f &ptRight = mpCurrFrame->mvrightpixel[i].pt;
            if (ptRight.x < 0) // 无效的双目匹配
                continue;

            float Parallax = ptLeft.x - ptRight.x;
            if (Parallax <= 0.0f)
                continue;

            float depth = static_cast<float>(fx * baseline / Parallax);

            if (depth <= 0.0f)
                continue;

            bool isFar = (depth > 40.0 * baseline);
            MapPoint::PointType type = isFar ? MapPoint::FAR : MapPoint::NEAR;
            // 归一化坐标 -> 世界坐标（初始位姿为单位矩阵）
            float u = (ptLeft.x - static_cast<float>(cx)) / static_cast<float>(fx);
            float v = (ptLeft.y - static_cast<float>(cy)) / static_cast<float>(fy);
            Eigen::Vector3d P_world = depth * Eigen::Vector3d(u, v, 1.0);

            // 构建 MapPoint
            cv::Mat position(3, 1, CV_32F);
            position.at<float>(0) = static_cast<float>(P_world.x());
            position.at<float>(1) = static_cast<float>(P_world.y());
            position.at<float>(2) = static_cast<float>(P_world.z());

            auto pMP = std::make_shared<MapPoint>(mNextMapPointId++, position, type);
            pMP->AddObservation(mpCurrFrame, i); // 添加观测
            mpMap->InsertMapPoint(pMP);          // 加入地图
            mpCurrFrame->mvpMapPoints[i] = pMP;  // 关联至当前帧
        }
        mbInitialized = true;
        // 将当前帧作为关键帧插入地图（供可视化用）
        mpMap->InsertKeyFrame(mpCurrFrame);
        std::cout << "[Tracker] Initial map generated with "
                  << mpCurrFrame->mvpMapPoints.size() << " points." << std::endl;
        return Eigen::Matrix4d::Identity();
    }
    if (mbInitialized && mpCurrFrame->mvleftpixel.size() > 0)
    {
        mpCurrFrame->mvpMapPoints.assign(mpCurrFrame->mvleftpixel.size(), nullptr);
        const int N = static_cast<int>(mpCurrFrame->mvleftpixel.size());
        const double fx = mpCamera->fx;
        const double fy = mpCamera->fy;
        const double cx = mpCamera->cx;
        const double cy = mpCamera->cy;
        const double baseline = mpCamera->mBaseline;

        // 遍历所有特征点
        for (int i = 0; i < N; ++i)
        {
            // 如果该点已经关联了地图点，跳过
            if (mpCurrFrame->mvpMapPoints[i] != nullptr)
                continue;

            const cv::Point2f &ptLeft = mpCurrFrame->mvleftpixel[i].pt;
            const cv::Point2f &ptRight = mpCurrFrame->mvrightpixel[i].pt;

            // 检查右图匹配是否有效
            if (ptRight.x < 0 || ptRight.y < 0)
                continue;

            // 计算视差（左图x - 右图x）
            float disparity = ptLeft.x - ptRight.x;
            if (disparity <= 0.0f)
                continue;

            // 计算深度
            float depth = static_cast<float>(fx * baseline / disparity);
            if (depth <= 0.0f)
                continue;

            // 判断远/近
            MapPoint::PointType type = (depth > 40.0 * baseline) ? MapPoint::FAR : MapPoint::NEAR;

            // 归一化坐标 -> 世界坐标（当前帧位姿为单位阵，简化）
            // 实际应使用当前帧的位姿将点变换到世界系，这里我们假定相机位于原点
            float u = (ptLeft.x - static_cast<float>(cx)) / static_cast<float>(fx);
            float v = (ptLeft.y - static_cast<float>(cy)) / static_cast<float>(fy);
            Eigen::Vector3d P_cam(u * depth, v * depth, depth); // 相机坐标系
            // 如果当前帧有真实位姿，需转换到世界系：
            // Eigen::Matrix4d Tcw = mpCurrFrame->GetPose().cast<double>();
            // Eigen::Vector3d P_world = Tcw.inverse().block<3,3>(0,0) * P_cam + Tcw.inverse().block<3,1>(0,3);
            // 这里简化，直接使用相机坐标作为世界坐标（假设世界系与第一帧相机系重合）

            cv::Mat position(3, 1, CV_32F);
            position.at<float>(0) = static_cast<float>(P_cam.x());
            position.at<float>(1) = static_cast<float>(P_cam.y());
            position.at<float>(2) = static_cast<float>(P_cam.z());

            // 创建地图点
            auto pMP = std::make_shared<MapPoint>(mNextMapPointId++, position, type);
            pMP->AddObservation(mpCurrFrame, i);
            mpMap->InsertMapPoint(pMP);
            mpCurrFrame->mvpMapPoints[i] = pMP;
        }
        return Eigen::Matrix4d::Identity();
    }
    return Eigen::Matrix4d::Identity();
}
