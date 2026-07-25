#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
// 前向声明
class Config;
class Camera;
class Map;
class FeatureDetector;
class Tracker;
class Viewer;

class System {
public:
    // 传感器类型，方便扩展单目/双目/RGBD
    enum Sensor {
        MONOCULAR = 0,
        STEREO = 1,
        RGBD = 2
    };

public:
    /**
     * @brief 构造函数：初始化系统核心组件
     * @param strConfigFile 配置文件路径 (kitti_configXX.yaml)
     * @param sensor 传感器类型
     * @param bUseViewer 是否开启 Pangolin 3D 可视化界面
     */
    System(const std::string &strConfigFile, const Sensor sensor = STEREO, const bool bUseViewer = true);

    ~System();

    /**
     * @brief 双目图像处理入口接口
     * @param imLeft  左目图像
     * @param imRight 右目图像
     * @param timestamp 时间戳 (秒)
     * @return 当前帧估计的位姿 Tcw (4x4 变换矩阵)
     */
    Eigen::Matrix4f TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp);

    /**
     * @brief 标记系统为结束状态，安全释放多线程
     */
    void Shutdown();

    /**
     * @brief 获取全局地图指针 (供外部获取轨迹/地图)
     */
    std::shared_ptr<Map> GetMap() const { return mpMap; }

private:
    Sensor mSensor;
    bool mbUseViewer;

    // 系统的核心组件实例
    std::shared_ptr<Camera> mpCamera0;        // 左目相机内参
    std::shared_ptr<Camera> mpCamera1;        // 右目相机内参
    std::shared_ptr<Map> mpMap;                // 全局地图 (存储 KeyFrames & MapPoints)
    std::shared_ptr<FeatureDetector> mpTrackerDetector; // 光流跟踪器
    std::shared_ptr<Tracker> mpTracker;        // 前端 Tracking 逻辑
    std::shared_ptr<Viewer> mpViewer;          // 3D 可视化渲染器

    // 可视化线程管理
    std::thread *mptViewer;
    
    std::mutex mMutexReset;
};

#endif // SYSTEM_H