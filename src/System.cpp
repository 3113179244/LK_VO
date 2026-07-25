#include "System.h"
#include "Config.h"
#include "Camera.h"
#include "Map.h"
#include "FeatureDetector.h"
#include "Tracker.h"
#include "Viewer.h"
#include <iostream>

System::System(const std::string &strConfigFile, const Sensor sensor, const bool bUseViewer)
    : mSensor(sensor), mbUseViewer(bUseViewer), mptViewer(nullptr)
{
    std::cout << "[System] Initializing Optical Flow VO System..." << std::endl;

    // 1. 加载参数文件
    if (!Config::setParameterFile(strConfigFile))
    {
        std::cerr << "[System] Fatal Error: Failed to load config file: " << strConfigFile << std::endl;
        exit(-1);
    }

    // 2. 实例化相机模型 (根据 Config 参数构建 Camera 对象)
    // 计算 baseline = ||body_T_cam1.col(3) - body_T_cam0.col(3)||
    double dBaseline = (Config::g_mBodyTCam1.block<3, 1>(0, 3) - Config::g_mBodyTCam0.block<3, 1>(0, 3)).norm();

    mpCamera0 = std::make_shared<Camera>(
        Config::g_dFx0, Config::g_dFy0, Config::g_dCx0, Config::g_dCy0,
        Config::g_dK1_0, Config::g_dK2_0, Config::g_dP1_0, Config::g_dP2_0, 0.0f,
        dBaseline);

    if (Config::g_nNumOfCam > 1)
    {
        mpCamera1 = std::make_shared<Camera>(
            Config::g_dFx1, Config::g_dFy1, Config::g_dCx1, Config::g_dCy1,
            Config::g_dK1_1, Config::g_dK2_1, Config::g_dP1_1, Config::g_dP2_1, 0.0f,
            dBaseline);
    }

    // 3. 初始化全局地图
    mpMap = std::make_shared<Map>();

    // 4. 初始化前端特征追踪与 Tracker
    mpTrackerDetector = std::make_shared<FeatureDetector>();

    // 初始化 Tracker 并注入依赖项 (Map, Camera, FeatureDetector)
    // 注：若你的 Tracker 构造函数参数不同，请自行调整
    // mpTracker = std::make_shared<Tracker>(mpMap, mpCamera0, mpTrackerDetector);

    // 5. 初始化 3D 可视化线程 (Pangolin)
    if (mbUseViewer)
    {
        mpViewer = std::make_shared<Viewer>(mpMap);
        mptViewer = new std::thread(&Viewer::Run, mpViewer);
        std::cout << "[System] Viewer thread started." << std::endl;
    }

    std::cout << "[System] System Initialization Complete!" << std::endl;
}

System::~System()
{
    Shutdown();
}

Eigen::Matrix4f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if (mSensor != STEREO)
    {
        std::cerr << "[System] Error: System is not initialized for Stereo processing!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 检查图像格式
    if (imLeft.empty() || imRight.empty())
    {
        std::cerr << "[System] Warning: Empty image received!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 1. 调用 FeatureDetector 运行光流追踪与特征提取
    mpTrackerDetector->TrackImage(timestamp, imLeft, imRight);

    // 2. 将数据交由 Tracker 运行位姿估计、初始化或帧间跟踪
    // Eigen::Matrix4f Tcw = mpTracker->GrabStereoImage(imLeft, imRight, timestamp, mpTrackerDetector);

    // 占位逻辑：目前先返回单位矩阵（或 Tracker 返回的位姿）
    Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();

    return Tcw;
}

void System::Shutdown()
{
    std::cout << "[System] System shutting down..." << std::endl;

    // 终止 Viewer 线程
    if (mpViewer)
    {
        mpViewer->RequestFinish();
        while (!mpViewer->isFinished())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    if (mptViewer)
    {
        if (mptViewer->joinable())
        {
            mptViewer->join();
        }
        delete mptViewer;
        mptViewer = nullptr;
    }

    std::cout << "[System] System stopped cleanly." << std::endl;
}