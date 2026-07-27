#ifndef MAP_H
#define MAP_H

#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include "Frame.h"
#include "MapPoint.h"

class Map {
public:
    typedef std::shared_ptr<Map> Ptr;

    Map();

    // ==========================================
    // 1. 数据插入与删除
    // ==========================================
    // 只有“关键帧”才会被插入地图，普通帧追踪完就丢弃
    void InsertKeyFrame(std::shared_ptr<Frame> frame);
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);
    
    // 清理质量差的地图点（例如重投影误差过大，或者不在当前视野内）
    void EraseMapPoint(std::shared_ptr<MapPoint> map_point);
    // 滑动窗口：剔除最老的一帧以维持计算规模
    void EraseOldestKeyFrame();

    // ==========================================
    // 2. 数据获取 (供后端优化和可视化调用)
    // ==========================================
    // 获取全局数据（如果你依然想保存所有轨迹）
    std::vector<std::shared_ptr<Frame>> GetAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();

    // 获取局部/活跃数据（光流里程计的核心，通常只取最近的 N 帧）
    std::vector<std::shared_ptr<Frame>> GetActiveKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetActiveMapPoints();

    // ==========================================
    // 3. 状态查询
    // ==========================================
    int GetKeyFramesInMap();
    int GetMapPointsInMap();

    // ==========================================
    // 4. 滑动窗口维护逻辑
    // ==========================================
    // 设置活动窗口的大小，例如只保留最近 10 帧
    void SetNumActiveKeyFrames(int num);

private:
    // ==========================================
    // 内部数据结构 (使用 unordered_map 以 ID 为 key 加速查找)
    // ==========================================
    std::unordered_map<unsigned long, std::shared_ptr<Frame>> mKeyFrames;
    std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> mMapPoints;

    // 活跃数据（滑动窗口）
    std::unordered_map<unsigned long, std::shared_ptr<Frame>> mActiveKeyFrames;
    std::unordered_map<unsigned long, std::shared_ptr<MapPoint>> mActiveMapPoints;

    int mNumActiveKeyframes; // 滑动窗口的最大帧数 (如 10 ~ 20)

    // ==========================================
    // 线程安全互斥锁 (极其关键)
    // ==========================================
    // 对 mKeyFrames 和 mMapPoints 的任何增删改查都必须加锁
    std::mutex mMutexMap;
};

#endif // MAP_H