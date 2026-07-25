#include "System.h"
#include "Config.h"
#include "Camera.h"
#include "Map.h"
#include "FeatureDetector.h"
#include "Tracker.h"
#include "Viewer.h"
#include <iostream>

System::System(const std::string &strConfigFile, const Sensor sensor, const bool bUseViewer)
    : mSensor(sensor), mbUseViewer(bUseViewer), mbShutdown(false)
{
    std::cout << "[System] Initializing Optical Flow VO System..." << std::endl;

    // 加载参数文件
    if (!Config::setParameterFile(strConfigFile))
    {
        std::cerr << "[System] Fatal Error: Failed to load config file: " << strConfigFile << std::endl;
        exit(-1);
    }

    // 实例化相机模型
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

    // 初始化地图与算法核心模块
    mpMap = std::make_shared<Map>();
    mpTrackerDetector = std::make_shared<FeatureDetector>();
    // mpTracker = std::make_shared<Tracker>(mpMap, mpCamera0, mpTrackerDetector);

    // 启动多线程管理
    StartThreads();

    std::cout << "[System] System Initialization Complete!" << std::endl;
}

System::~System()
{
    Shutdown();
}

void System::StartThreads()
{
    // 启动 Pangolin 3D 可视化渲染线程
    if (mbUseViewer)
    {
        mpViewer = std::make_shared<Viewer>(mpMap);
        mptViewer = std::make_unique<std::thread>(&Viewer::Run, mpViewer);
        std::cout << "[System] Viewer thread created & started." << std::endl;
    }

    // 如果后续加入后端优化线程，可以在此处统一启动：
    // mptBackend = std::make_unique<std::thread>(&Optimizer::RunLocalBA, mpOptimizer);
}

Eigen::Matrix4f System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if (IsShutDown())
    {
        std::cerr << "[System] Warning: System is shut down. Tracking aborted." << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    if (mSensor != STEREO)
    {
        std::cerr << "[System] Error: System is not initialized for Stereo processing!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    if (imLeft.empty() || imRight.empty())
    {
        std::cerr << "[System] Warning: Empty image received!" << std::endl;
        return Eigen::Matrix4f::Identity();
    }

    // 光流特征追踪
    mpTrackerDetector->TrackImage(timestamp, imLeft, imRight);

    // 位姿估算
    Eigen::Matrix4f Tcw = Eigen::Matrix4f::Identity();

    return Tcw;
}

void System::Shutdown()
{
    // 避免重复触发 Shutdown 逻辑
    bool bExpected = false;
    if (!mbShutdown.compare_exchange_strong(bExpected, true))
    {
        return;
    }

    std::cout << "[System] System shutting down..." << std::endl;

    // 终止 Viewer 渲染线程
    if (mpViewer)
    {
        mpViewer->RequestFinish();
    }

    if (mptViewer && mptViewer->joinable())
    {
        mptViewer->join();
        mptViewer.reset();
        std::cout << "[System] Viewer thread joined cleanly." << std::endl;
    }

    std::cout << "[System] System stopped cleanly." << std::endl;
}