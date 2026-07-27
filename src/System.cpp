#include "System.h"
#include "Config.h"
#include "Camera.h"
#include "Map.h"
#include "Tracker.h"
#include "Viewer.h"
#include <iostream>
System::System(const std::string &strConfigFile, const eSensor sensor, const bool bUseViewer)
    : mSensor(sensor), mbUseViewer(bUseViewer), mbShutdown(false)
{

    // 加载配置文件
    Config::setParameterFile(strConfigFile);

    // 初始化 Tracking 核心追踪器
    mpTracker = std::make_shared<Tracker>(mpCamera, mpMap);

    // 初始化 Viewer 并启动子线程
    if (mbUseViewer)
    {
        mpViewer = std::make_shared<Viewer>(mpMap);
        StartThreads();
    }

    std::cout << "System Initialization Complete." << std::endl;
}

System::~System()
{
    if (!mbShutdown.load())
    {
        Shutdown();
    }
}

Eigen::Matrix4d System::TrackStereo(const double &timestamp, const cv::Mat &image0, const cv::Mat &image1)
{
    if (mSensor != STEREO)
    {
        std::cerr << "错误: 当前系统未设置为 STEREO 双目模式！" << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    if (mbShutdown.load())
    {
        std::cerr << "系统已关闭，拒绝处理新图像。" << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    // 调用 Tracker 进行位姿估计 (Tracker::GrabImageStereo 返回 Matrix4d)
    Eigen::Matrix4d Tcw = mpTracker->GrabImageStereo(timestamp, image0, image1);

    return Tcw;
}

void System::StartThreads()
{
    if (mbUseViewer && mpViewer)
    {
        mpViewerThread = std::make_unique<std::thread>(&Viewer::Run, mpViewer);
        std::cout << "[System] Viewer thread started successfully." << std::endl;
    }
}

void System::Shutdown()
{
    // 防止重复执行关闭逻辑
    bool expected = false;
    if (!mbShutdown.compare_exchange_strong(expected, true))
    {
        return;
    }

    std::cout << "\n[System] Shutting down SLAM System..." << std::endl;

    // 请求可视化线程退出
    if (mbUseViewer && mpViewer)
    {
        mpViewer->RequestFinish();
        if (mpViewerThread && mpViewerThread->joinable())
        {
            mpViewerThread->join();
            std::cout << "[System] Viewer thread joined." << std::endl;
        }
    }

    std::cout << "[System] System shutdown complete." << std::endl;
}