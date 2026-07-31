#include "Tracker.h"
#include "Camera.h"
#include "Map.h"
#include "MapPoint.h"
#include <iostream>

// 实现构造函数
Tracker::Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap)
    : mpCamera(pCamera), mpMap(pMap),
      mbInitialized(true), mNextMapPointId(0), mNextFrameId(0)
{
    mpFeatureDetector = std::make_shared<FeatureDetector>();
    std::cout << "[Tracker] Initialized successfully." << std::endl;
}

Eigen::Matrix4d Tracker::GrabImageStereo(const double timestamp, const cv::Mat &image0, const cv::Mat &image1, cv::Mat &matDisplay)
{
    mpCurrFrame = std::make_shared<Frame>(timestamp, image0, image1, mNextFrameId++);
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
    if (mbInitialized)
    {
        const int N = static_cast<int>(mpCurrFrame->mvleftpixel.size());
        double fx = mpCamera->fx;
        double fy = mpCamera->fy;
        double cx = mpCamera->cx;
        double cy = mpCamera->cy;
        double baseline = mpCamera->mBaseline;

        mpCurrFrame->mvpMapPoints.resize(N, nullptr);

        for (int i = 0; i < N; ++i)
        {
            const cv::Point2f &ptLeft = mpCurrFrame->mvleftpixel[i].pt;
            const cv::Point2f &ptRight = mpCurrFrame->mvrightpixel[i].pt;
            if (ptRight.x < 0)
                continue;

            float Parallax = ptLeft.x - ptRight.x;
            if (Parallax <= 0.0f)
                continue;

            float depth = static_cast<float>(fx * baseline / Parallax);
            if (depth <= 0.0f)
                continue;

            bool isFar = (depth > 40.0 * baseline);
            MapPoint::PointType type = isFar ? MapPoint::FAR : MapPoint::NEAR;

            float u = (ptLeft.x - static_cast<float>(cx)) / static_cast<float>(fx);
            float v = (ptLeft.y - static_cast<float>(cy)) / static_cast<float>(fy);
            Eigen::Vector3d P_world = depth * Eigen::Vector3d(u, v, 1.0);

            cv::Mat position(3, 1, CV_32F);
            position.at<float>(0) = static_cast<float>(P_world.x());
            position.at<float>(1) = static_cast<float>(P_world.y());
            position.at<float>(2) = static_cast<float>(P_world.z());

            auto pMP = std::make_shared<MapPoint>(mNextMapPointId++, position, type);
            pMP->AddObservation(mpCurrFrame, i);
            mpMap->InsertMapPoint(pMP);
            mpCurrFrame->mvpMapPoints[i] = pMP;
        }
        mpCurrFrame->SetPose(Eigen::Matrix4f::Identity());
        mpLastKeyFrame = mpCurrFrame;
        mNumFramesSinceLastKeyFrame = 0;
        mpMap->InsertKeyFrame(mpCurrFrame);
        mbInitialized = false;
        std::cout << "[Tracker] Initial map generated with "
                  << mpCurrFrame->mvpMapPoints.size() << " points." << std::endl;
        return mpCurrFrame->GetPose().cast<double>();
    }
    if (!mbInitialized)
    {
        const int N = static_cast<int>(mpCurrFrame->mvleftpixel.size());
        const double fx = mpCamera->fx;
        const double fy = mpCamera->fy;
        const double cx = mpCamera->cx;
        const double cy = mpCamera->cy;
        const double baseline = mpCamera->mBaseline;

        // 收集已有地图点（NEAR）用于PnP
        std::vector<cv::Point3f> pts3d;
        std::vector<cv::Point2f> pts2d;
        int nearCount = 0, farCount = 0, badCount = 0, nullCount = 0;
        for (int i = 0; i < N; ++i)
        {
            auto pMP = mpCurrFrame->mvpMapPoints[i];
            if (pMP)
            {
                if (pMP->isBad())
                    badCount++;
                else if (pMP->GetType() == MapPoint::NEAR)
                {
                    nearCount++;
                    cv::Mat pos = pMP->GetMapPoints();
                    pts3d.push_back(cv::Point3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2)));
                    pts2d.push_back(mpCurrFrame->mvleftpixel[i].pt);
                }
                else
                    farCount++;
            }
            else
            {
                nullCount++;
            }
        }
        std::cout << "[Tracker] Frame " << mpCurrFrame->mFrameId
                  << " | NEAR: " << nearCount
                  << " | FAR: " << farCount
                  << " | BAD: " << badCount
                  << " | NULL: " << nullCount << std::endl;

        // PnP 估计当前帧位姿
        if (pts3d.size() >= 10)
        {
            cv::Mat K = (cv::Mat_<double>(3, 3) << mpCamera->fx, 0, mpCamera->cx,
                         0, mpCamera->fy, mpCamera->cy,
                         0, 0, 1);
            cv::Mat rvec, tvec, inliers;
            cv::solvePnPRansac(pts3d, pts2d, K, cv::Mat(), rvec, tvec,
                               false, 100, 4.0, 0.99, inliers);

            if (inliers.rows >= 8)
            {
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                Eigen::Matrix4d Tcw;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        Tcw(i, j) = R.at<double>(i, j);
                Tcw(0, 3) = tvec.at<double>(0);
                Tcw(1, 3) = tvec.at<double>(1);
                Tcw(2, 3) = tvec.at<double>(2);
                Tcw(3, 3) = 1.0;
                mpCurrFrame->SetPose(Tcw.cast<float>());
                std::cout << "Frame " << mpCurrFrame->mFrameId << " PnP success pose:\n"
                          << mpCurrFrame->GetPose().matrix() << std::endl;
            }
            else
            {
                if (mpPrevFrame)
                    mpCurrFrame->SetPose(mpPrevFrame->GetPose());
                else
                    mpCurrFrame->SetPose(Eigen::Matrix4f::Identity());
            }
        }
        else
        {
            if (mpPrevFrame)
                mpCurrFrame->SetPose(mpPrevFrame->GetPose());
            else
                mpCurrFrame->SetPose(Eigen::Matrix4f::Identity());
            std::cerr << "[Tracker] Warning: Insufficient 3D-2D matches for PnP (found "
                      << pts3d.size() << " points). Using previous frame pose." << std::endl;
        }

        // 判断是否需要新的关键帧
        bool bNeedNewKF = false;
        if (mNumFramesSinceLastKeyFrame >= 20)
        {
            bNeedNewKF = true;
        }
        else
        {
            int nCurr = mpCurrFrame->mvleftpixel.size();
            int nRef = mpLastKeyFrame ? mpLastKeyFrame->mvleftpixel.size() : 0;
            double ratio = (nRef > 0) ? (double)nCurr / nRef : 0.0;
            if (ratio < 0.8)
            {
                bNeedNewKF = true;
            }
        }

        // 如果是关键帧，则生成新的地图点
        if (bNeedNewKF)
        {
            // 确保 mvpMapPoints 大小与特征点一致
            if (static_cast<int>(mpCurrFrame->mvpMapPoints.size()) != N)
                mpCurrFrame->mvpMapPoints.resize(N, nullptr);

            for (int i = 0; i < N; ++i)
            {
                // 跳过已关联地图点的特征
                if (mpCurrFrame->mvpMapPoints[i] != nullptr)
                    continue;

                const cv::Point2f &ptLeft = mpCurrFrame->mvleftpixel[i].pt;
                const cv::Point2f &ptRight = mpCurrFrame->mvrightpixel[i].pt;
                if (ptRight.x < 0 || ptRight.y < 0)
                    continue;

                float disparity = ptLeft.x - ptRight.x;
                if (disparity <= 0.0f)
                    continue;

                float depth = static_cast<float>(fx * baseline / disparity);
                if (depth <= 0.0f)
                    continue;

                MapPoint::PointType type = (depth > 40.0 * baseline) ? MapPoint::FAR : MapPoint::NEAR;

                // 相机坐标系下的点
                float u = (ptLeft.x - static_cast<float>(cx)) / static_cast<float>(fx);
                float v = (ptLeft.y - static_cast<float>(cy)) / static_cast<float>(fy);
                Eigen::Vector3d P_cam(u * depth, v * depth, depth);

                // 利用当前帧位姿转换到世界坐标系
                Eigen::Matrix4f Tcw_f = mpCurrFrame->GetPose();
                Eigen::Matrix4d Tcw = Tcw_f.cast<double>();
                Eigen::Vector4d P_cam_homo(P_cam.x(), P_cam.y(), P_cam.z(), 1.0);
                Eigen::Vector4d P_world_homo = Tcw.inverse() * P_cam_homo;
                Eigen::Vector3d P_world = P_world_homo.head<3>() / P_world_homo.w();

                cv::Mat position(3, 1, CV_32F);
                position.at<float>(0) = static_cast<float>(P_world.x());
                position.at<float>(1) = static_cast<float>(P_world.y());
                position.at<float>(2) = static_cast<float>(P_world.z());

                auto pMP = std::make_shared<MapPoint>(mNextMapPointId++, position, type);
                pMP->AddObservation(mpCurrFrame, i);
                mpMap->InsertMapPoint(pMP);
                mpCurrFrame->mvpMapPoints[i] = pMP;
            }

            // 插入关键帧并重置计数器
            mpMap->InsertKeyFrame(mpCurrFrame);
            mpLastKeyFrame = mpCurrFrame;
            mNumFramesSinceLastKeyFrame = 0;
        }
        else
        {
            // 普通帧：不生成新点，仅增加帧计数
            mNumFramesSinceLastKeyFrame++;
        }

        return mpCurrFrame->GetPose().cast<double>();
    }
}