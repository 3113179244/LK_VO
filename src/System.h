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
class Camera;
class Map;
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
    System(const std::string &strConfigFile, const eSensor sensor = STEREO, const bool bUseViewer = true);
    ~System();

    Eigen::Matrix4d TrackStereo(const double &timestamp, const cv::Mat &image0, const cv::Mat &image1);

    void Shutdown();
    bool IsShutDown() const { return mbShutdown.load(); }

private:
    void StartThreads();

private:
    eSensor mSensor;
    bool mbUseViewer;
    std::atomic<bool> mbShutdown;
    std::shared_ptr<Camera> mpCamera;
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Tracker> mpTracker;
    std::shared_ptr<Viewer> mpViewer;
    std::unique_ptr<std::thread> mpViewerThread;
};

#endif // SYSTEM_H