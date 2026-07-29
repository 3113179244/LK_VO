#include "MapPoint.h"
#include "Frame.h"

/**
 * @brief 构造函数实现
 * @param id     地图点ID
 * @param points 初始三维坐标
 *
 * 初始化坐标（克隆一份独立数据），观测计数为0，坏点标志为false。
 */
MapPoint::MapPoint(const long unsigned int id, const cv::Mat &points, PointType type)
    : mId(id), mMapPoints(points.clone()), mnObs(0), mbBad(false), mType(type) {}

/**
 * @brief 线程安全地更新三维坐标
 * @param points 新坐标
 *
 * 使用互斥锁保护，并复制数据到内部矩阵。
 */
void MapPoint::SetMapPoints(const cv::Mat &points)
{
    std::unique_lock<std::mutex> lock(mMutexMapPoints);
    points.copyTo(mMapPoints);
}

/**
 * @brief 线程安全地获取三维坐标
 * @return 坐标矩阵的深拷贝，确保外部修改不影响内部数据
 */
cv::Mat MapPoint::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMapPoints);
    return mMapPoints.clone();
}

/**
 * @brief 添加观测关系
 * @param pFrame 观测关键帧
 * @param idx    特征索引
 *
 * 若该帧已存在于观测列表中，则忽略；否则插入并增加计数。
 * 操作受互斥锁保护，确保线程安全。
 */
void MapPoint::AddObservation(std::shared_ptr<Frame> pFrame, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pFrame))
        return; // 已存在，不重复添加
    mObservations[pFrame] = idx;
    mnObs++;
}

/**
 * @brief 移除一个观测
 * @param pFrame 要移除的关键帧
 *
 * 若该帧存在，则从映射中删除并减少计数。
 */
void MapPoint::RemoveObservation(std::shared_ptr<Frame> pFrame)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pFrame))
    {
        mObservations.erase(pFrame);
        mnObs--;
    }
}

/**
 * @brief 获取所有观测的副本
 * @return 包含所有观测的映射副本（线程安全）
 */
std::map<std::shared_ptr<Frame>, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations; // 返回拷贝，外部修改不影响内部
}

/**
 * @brief 获取观测数量
 * @return 观测帧个数
 */
int MapPoint::GetObservedCount()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mnObs;
}

/**
 * @brief 将地图点标记为坏点
 *
 * 设置mbBad为true，后续可被地图清理线程删除。
 * 操作受互斥锁保护。
 */
void MapPoint::SetBadFlag()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mbBad = true;
}

/**
 * @brief 检查地图点是否为坏点
 * @return true 表示坏点
 */
bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mbBad;
}

MapPoint::PointType MapPoint::GetType() 
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mType;
}

void MapPoint::SetType(PointType type)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mType = type;
}