#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include "Frame.h"
#include "Camera.h"
#include "Map.h"

class KeyFrame
{
public:
    typedef std::shared_ptr<KeyFrame> Ptr;

    // 构造函数：基于一个普通 Frame 来实例化一个 KeyFrame
    KeyFrame(std::shared_ptr<Frame> pFrame);

    // ==========================================
    // 1. 位姿读写 (后端 BA 会不断修改它的位姿)
    // ==========================================
    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();
    Eigen::Vector3f GetCameraCenter();

    // ==========================================
    // 2. 地图点关联 (MapPoint Observation)
    // ==========================================
    // 获取该关键帧能看到的所有 3D 地图点
    std::vector<std::shared_ptr<MapPoint>> GetMapPoints();

    // 获取特定特征点索引对应的 MapPoint
    std::shared_ptr<MapPoint> GetMapPoint(const size_t idx);

    // 建立/解除 特征点索引 与 MapPoint 的关联
    void AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t idx);
    void EraseMapPointMatch(const size_t idx);
    void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);

    // ==========================================
    // 3. 共视关系 (Covisibility Graph)
    // ==========================================
    // 在纯光流滑动窗口中，通常根据时间顺序和共视点数量来建立连接
    void AddConnection(std::shared_ptr<KeyFrame> pKF, const int weight);
    void EraseConnection(std::shared_ptr<KeyFrame> pKF);
    // 获取与当前关键帧共视的所有其他关键帧及权重(共视点数量)
    std::map<std::shared_ptr<KeyFrame>, int> GetConnectedKeyFrames();

public:
    // ==========================================
    // 基础信息 (元数据)
    // ==========================================
    const long unsigned int mId;      // 关键帧的全局唯一ID (注意: 不是Frame的ID)
    const long unsigned int mFrameId; // 对应的原始 Frame ID
    const double mTimeStamp;          // 时间戳

    // ==========================================
    // 传感器信息 (通常从 Frame 深拷贝过来)
    // ==========================================
    std::shared_ptr<Camera> mpCamera;
    const float mbf;
    const int N; // 特征点总数

    // 2D 特征点观测 (从 Frame 拷贝，用于计算重投影误差)
    // 在光流法中，这里存储的是通过光流追踪到的像素坐标
    std::vector<cv::KeyPoint> mvKeysLeft;
    std::vector<int> mvFeatureIds; // 对应的全局光流特征点ID

private:
    // ==========================================
    // 状态数据与线程锁
    // ==========================================
    Eigen::Matrix4f mTcw; // 世界到相机的变换矩阵 $T_{cw}$
    Eigen::Vector3f mOw;  // 相机光心在世界坐标系下的坐标

    // 存储当前帧关联的 MapPoint (长度等于 N，未关联则为 nullptr)
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;

    // 共视关系记录
    std::map<std::shared_ptr<KeyFrame>, int> mConnectedKeyFrameWeights;

    // 多线程互斥锁
    std::mutex mMutexPose;        // 保护位姿 $T_{cw}$
    std::mutex mMutexFeatures;    // 保护 MapPoints 的增删改
    std::mutex mMutexConnections; // 保护共视关系的更新
};

#endif