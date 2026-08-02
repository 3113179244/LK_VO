#ifndef SYSTEM_H
#define SYSTEM_H

#include <iostream>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 前向声明
class Map;
class Tracker;
class Viewer;
class ORBVocabulary;
class FrameDrawer;
class System
{
public:
    // 传感器模式枚举
    enum eSensor
    {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:
    System(const std::string &strConfigFile, const eSensor sensor = STEREO, const bool bUseViewer = true);
    ~System();
    cv::Mat DrawFrame();
    std::shared_ptr<FrameDrawer> GetFrameDrawer() const { return mpFrameDrawer; }
    // 核心输入接口：传入左右目图像和时间戳，返回世界到相机的变换 Tcw
    Eigen::Matrix4f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    // 控制接口
    void Reset();
    void Shutdown();

    // 数据获取接口
    std::shared_ptr<Map> GetMap() const { return mpMap; }

private:
    eSensor mSensor;

    // 系统三大核心模块/数据结构
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;

    // 可视化与后台线程句柄
    std::thread *mpViewerThread;
    std::shared_ptr<FrameDrawer> mpFrameDrawer;
    // 线程安全互斥锁
    std::mutex mMutexMode;
};

#endif // SYSTEM_H