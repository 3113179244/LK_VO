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
    enum Sensor {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:
    System(const std::string &strConfigFile, const Sensor sensor = STEREO, const bool bUseViewer = true);
    ~System();

    // 跟踪接口
    Eigen::Matrix4f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    // 线程管理核心接口
    void Shutdown();
    bool IsShutDown() const { return mbShutdown.load(); }

    std::shared_ptr<Map> GetMap() const { return mpMap; }

private:
    // 初始化子线程
    void StartThreads();

private:
    Sensor mSensor;
    bool mbUseViewer;
    std::atomic<bool> mbShutdown; // 原子类型标志位，保证多线程状态同步

    // SLAM 核心模块组件
    std::shared_ptr<Camera> mpCamera0;
    std::shared_ptr<Camera> mpCamera1;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<FeatureDetector> mpTrackerDetector;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;

    // 线程管理对象 (使用 std::unique_ptr 智能指针实现 RAII)
    std::unique_ptr<std::thread> mptViewer;
    // std::unique_ptr<std::thread> mptBackend; // 预留后端优化线程
    
    mutable std::mutex mMutexReset;
};

#endif // SYSTEM_H