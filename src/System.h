#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

// 前向声明
class Config;
class Camera;
class Map;
class FeatureDetector;
class Tracker;
class Viewer;

class System {
public:
    enum eSensor {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:
    // 适配 main.cpp 的调用，默认传感器类型为 STEREO
    System(const std::string &strConfigFile, const eSensor sensor = STEREO, const bool bUseViewer = true);
    ~System();

    // 双目跟踪接口（转接至 Tracker）
    Eigen::Matrix4d TrackStereo(const double &timestamp, const cv::Mat &image0, const cv::Mat &image1);

    // 线程管理核心接口
    void Shutdown();
    bool IsShutDown() const { return mbShutdown.load(); }

private:
    void StartThreads();

private:
    eSensor mSensor;
    bool mbUseViewer;
    std::atomic<bool> mbShutdown; // 原子类型标志位，保证多线程状态同步
    std::shared_ptr<Camera> mpCamera;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;
    std::unique_ptr<std::thread> mpViewerThread;
};

#endif // SYSTEM_H