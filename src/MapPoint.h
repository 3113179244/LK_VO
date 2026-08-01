#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <vector>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class KeyFrame;
class Frame;
class Map;

class MapPoint
{
public:
    // 从双目/深度 Frame 构造点
    MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap);
    // 从两个 KeyFrame 三角化构造点
    MapPoint(const Eigen::Vector3f &Pos, Map* pMap, KeyFrame* pRefKF, const int &idxF);

    // ---- 世界坐标管理 (线程安全) ----
    void SetWorldPos(const Eigen::Vector3f &Pos);
    Eigen::Vector3f GetWorldPos();
    Eigen::Vector3f GetNormal(); // 获取平均观测方向

    // ---- 观测管理 (Observations) ----
    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);
    std::map<KeyFrame*, size_t> GetObservations();
    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    // ---- ORB 特征属性维护 ----
    void ComputeDistinctiveDescriptor(); // 计算代表性描述子
    cv::Mat GetDescriptor();
    void UpdateNormalAndDepth();         // 更新平均法线方向与深度范围

    // ---- 状态与坏点剔除 (Bad Point Tracking) ----
    void SetBadFlag();
    bool isBad();
    void Replace(MapPoint* pMP);         // 替换/融合地图点

    // ---- 追踪过程统计 (用于局部地图过滤) ----
    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();

    static long unsigned int nNextId;
    long unsigned int mnId;
    static std::mutex mGlobalMutex;

    // 统计变量 (用于Tracking与Local Mapping剔除坏点)
    int mnVisible;
    int mnFound;

    // 标记与状态
    bool mbBad;
    MapPoint* mpReplaced;

    // 视角与距离限制 (用于特征匹配与重投影)
    float mfMinDistance;
    float mfMaxDistance;

private:
    // ---- 线程安全数据 ----
    std::mutex mMutexPos;
    std::mutex mMutexFeatures;

    Eigen::Vector3f mWorldPos;           // 3D 位置 (World 坐标系)
    std::map<KeyFrame*, size_t> mObservations; // 观测到该点的 KeyFrame 及对应的特征点 Index
    Eigen::Vector3f mNormalVector;       // 平均观测方向（单位向量）
    cv::Mat mDescriptor;                 // 代表性描述子 (最接近中位数的描述子)

    // 引用关键帧与地图
    KeyFrame* mpRefKF;
    Map* mpMap;
};

#endif // MAPPOINT_H