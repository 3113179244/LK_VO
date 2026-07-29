#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/opencv.hpp>
#include <map>
#include <memory>
#include <mutex>

// 前向声明，避免循环包含
class Frame;

/**
 * @brief 地图点类，表示SLAM系统中的3D地图点
 *
 * 存储三维世界坐标，并维护所有观测到该点的关键帧及对应特征索引，
 * 同时提供线程安全的访问接口。
 */
class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint> Ptr; // 智能指针类型别名
    enum PointType
    {
        NEAR = 0,
        FAR = 1
    };
    /**
     * @brief 构造函数
     * @param id     地图点唯一ID
     * @param points 三维坐标（通常为3x1的CV_32F矩阵）
     */
    MapPoint(const long unsigned int id, const cv::Mat &points, PointType type = NEAR);
    PointType GetType();
    void SetType(PointType type);
    /**
     * @brief 设置三维坐标（线程安全）
     * @param points 新的坐标矩阵
     */
    void SetMapPoints(const cv::Mat &points);

    /**
     * @brief 获取三维坐标的副本（线程安全）
     * @return 坐标矩阵的深拷贝
     */
    cv::Mat GetMapPoints();

    /**
     * @brief 添加一个观测（某个关键帧在特定特征索引处观测到该点）
     * @param pFrame 观测关键帧的智能指针
     * @param idx    该帧中对应特征的索引
     * @note 若该帧已存在则不会重复添加
     */
    void AddObservation(std::shared_ptr<Frame> pFrame, size_t idx);

    /**
     * @brief 移除一个观测
     * @param pFrame 要移除的关键帧
     */
    void RemoveObservation(std::shared_ptr<Frame> pFrame);

    /**
     * @brief 获取所有观测信息（线程安全）
     * @return 映射：关键帧 -> 特征索引
     */
    std::map<std::shared_ptr<Frame>, size_t> GetObservations();

    /**
     * @brief 获取观测数量
     * @return 观测帧的个数
     */
    int GetObservedCount();

    /**
     * @brief 将地图点标记为“坏点”（不再使用）
     */
    void SetBadFlag();

    /**
     * @brief 检查地图点是否被标记为坏点
     * @return true 表示坏点
     */
    bool isBad();

    const long unsigned int mId; ///< 地图点ID，只读（初始化后不可变）

private:
    cv::Mat mMapPoints;                                     ///< 三维坐标（世界坐标系）
    std::map<std::shared_ptr<Frame>, size_t> mObservations; ///< 观测信息
    int mnObs;                                              ///< 观测数量（冗余变量，与mObservations.size()一致）
    bool mbBad;                                             ///< 坏点标志
    PointType mType;   // 远/近标志
    std::mutex mMutexMapPoints; ///< 保护mMapPoints的互斥锁
    std::mutex mMutexFeatures;  ///< 保护mObservations, mnObs, mbBad的互斥锁
};

#endif // MAPPOINT_H