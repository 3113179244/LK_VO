#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

// 前向声明，避免循环包含
class Frame;
class Camera;
class Map;
class MapPoint;

/**
 * @brief 关键帧类，代表SLAM系统中被选为关键帧的普通帧
 * 
 * 关键帧存储了位姿、关联的地图点以及与其它关键帧的共视关系，
 * 是建图和闭环检测的基本单元。
 */
class KeyFrame
{
public:
    typedef std::shared_ptr<KeyFrame> Ptr;   // 智能指针类型别名

    /**
     * @brief 从普通帧构造关键帧，拷贝必要的数据
     * @param pFrame 普通帧的共享指针
     */
    KeyFrame(std::shared_ptr<Frame> pFrame);

    // ---------- 位姿相关 ----------
    void SetPose(const Eigen::Matrix4f &Tcw);   // 设置相机到世界的变换矩阵（Tcw）
    Eigen::Matrix4f GetPose();                  // 获取当前位姿（线程安全）
    Eigen::Vector3f GetCameraCenter();          // 获取相机光心在世界坐标系中的坐标

    // ---------- 地图点管理 ----------
    std::vector<std::shared_ptr<MapPoint>> GetMapPoints();          // 获取所有关联的地图点
    std::shared_ptr<MapPoint> GetMapPoint(const size_t idx);        // 获取指定索引的地图点

    void AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t idx); // 在指定索引处添加地图点
    void EraseMapPointMatch(const size_t idx);                         // 删除指定索引的地图点匹配
    void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);            // 删除所有指向该地图点的匹配

    // ---------- 共视关系管理 ----------
    void AddConnection(std::shared_ptr<KeyFrame> pKF, const int weight);  // 添加或增加共视权重
    void EraseConnection(std::shared_ptr<KeyFrame> pKF);                  // 移除与某关键帧的共视关系
    std::map<std::shared_ptr<KeyFrame>, int> GetConnectedKeyFrames();     // 获取所有共视关键帧及权重

    // ---------- 只读成员变量 ----------
    const long unsigned int mId;      // 全局唯一ID（由静态计数器生成）
    const long unsigned int mFrameId; // 原始普通帧的ID
    const double mTimeStamp;          // 时间戳

    std::shared_ptr<Camera> mpCamera; // 关联的相机模型（可能为空，待外部设置）
    const int N;                      // 特征点总数（来自原始帧）

    std::vector<cv::KeyPoint> mvKeysLeft; // 左目（或单目）特征点
    std::vector<int> mvFeatureIds;        // 特征点对应的全局特征ID（可能用于跟踪）

private:
    // ---------- 位姿数据 ----------
    Eigen::Matrix4f mTcw;   // 世界到相机的变换矩阵（Twc的逆，即相机位姿）
    Eigen::Vector3f mOw;   // 相机光心坐标（世界系）

    // ---------- 地图点数据 ----------
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints; // 与特征点一一对应的地图点

    // ---------- 共视关系 ----------
    std::map<std::shared_ptr<KeyFrame>, int> mConnectedKeyFrameWeights; // 关键帧→共视权重

    // ---------- 线程锁 ----------
    std::mutex mMutexPose;       // 保护位姿
    std::mutex mMutexFeatures;   // 保护地图点
    std::mutex mMutexConnections;// 保护共视关系

    static long unsigned int nNextId; // 用于生成唯一ID的静态计数器
};

#endif // KEYFRAME_H