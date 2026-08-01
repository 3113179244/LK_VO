#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <vector>
#include <set>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

// 前向声明
class Frame;
class MapPoint;
class Map;
class ORBVocabulary;

class KeyFrame
{
public:
    // 从 Frame 构造 KeyFrame
    KeyFrame(Frame &F, Map* pMap);

    // ---- 位姿管理 (支持多线程安全，使用 Eigen 替代 cv::Mat) ----
    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();
    Eigen::Matrix4f GetPoseInverse();
    Eigen::Vector3f GetCameraCenter();
    Eigen::Matrix3f GetRotation();
    Eigen::Vector3f GetTranslation();

    // ---- 共视图 (Covisibility Graph) 管理 ----
    void AddConnection(KeyFrame* pKF, const int &weight);
    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::vector<KeyFrame*> GetCovisibleByWeight(const int &w);
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
    std::vector<KeyFrame*> GetConnectedKeyFrames();
    int GetWeight(KeyFrame* pKF);

    // ---- 地图点关联 ----
    void AddMapPoint(MapPoint* pMP, const size_t &idx);
    void EraseMapPointMatch(const size_t &idx);
    void EraseMapPointMatch(MapPoint* pMP);
    void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
    std::vector<MapPoint*> GetMapPointMatches();
    MapPoint* GetMapPoint(const size_t &idx);

    // ---- 词袋与属性检索 ----
    void ComputeBoW();

public:
    // 标识与时间戳
    static long unsigned int nNextId;
    long unsigned int mnId;
    const long unsigned int mnFrameId; // 对应的原始 Frame ID
    const double mTimeStamp;

    // 相机内参及双目参数
    const float fx, fy, cx, cy, invfx, invfy;
    const float mbf, mb, mThDepth;
    const cv::Mat mK;

    // 特征点与描述子 (只保存左图/主的)
    const int N;
    const std::vector<cv::KeyPoint> mvKeys;
    const std::vector<cv::KeyPoint> mvKeysUn;
    const std::vector<float> mvuRight;
    const std::vector<float> mvDepth;
    const cv::Mat mDescriptors;

    // 状态与图标记
    bool mbBad; // 标记是否被冗余删除剔除

protected:
    // ---- 线程安全的数据保护 ----
    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;

    // 相机位姿 (World -> Camera) - 改用 Eigen 存储
    Eigen::Matrix4f Tcw;
    Eigen::Vector3f Ow;
    Eigen::Matrix3f Rcw;
    Eigen::Vector3f tcw;
    Eigen::Matrix3f Rwc;

    // 关联的地图点
    std::vector<MapPoint*> mvpMapPoints;

    // 共视图数据结构：记录相连的关键帧及其权重（共享地图点数）
    std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
    std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
    std::vector<int> mvOrderedWeights;

    // 关联的地图指针与字典指针
    Map* mpMap;
    ORBVocabulary* mpORBvocabulary;
};

#endif // KEYFRAME_H