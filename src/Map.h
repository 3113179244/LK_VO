#ifndef MAP_H
#define MAP_H

#include <memory>
#include <unordered_map>
#include <mutex>
#include <limits>
#include "MapPoint.h"
#include "Frame.h"

class Map
{
public:
    typedef std::shared_ptr<Map> Ptr;

    // 数据类型定义：通过 ID 索引对象
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

    Map() = default;

    // --- 核心 API ---

    // 1. 插入地图点与关键帧
    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);

    // 2. 获取所有地图点与关键帧（常用于 Viewer 绘图和 Optimizer 全局优化）
    LandmarksType getAllMapPoints();
    KeyframesType getAllKeyFrames();

    // 3. 获取活跃的地图点与关键帧（局部地图，用于前端 LK 跟踪）
    LandmarksType getActiveMapPoints();
    KeyframesType getActiveKeyFrames();

    // 4. 清理无效/离群地图点 (Outliers)
    void cleanMap();

    // 5. 移除指定的地图点
    void removeMapPoint(MapPoint::Ptr map_point);

private:
    std::mutex data_mutex_; // 数据保护互斥锁

    // 全局地图点与关键帧集合
    LandmarksType landmarks_;        // <MapPoint_ID, MapPoint_Ptr>
    LandmarksType active_landmarks_; // 局部/活跃地图点

    KeyframesType keyframes_;        // <Frame_ID, Frame_Ptr>
    KeyframesType active_keyframes_; // 局部/活跃关键帧

    // 局部地图的最大容量控制（如保持最近的 7 个关键帧）
    int num_active_keyframes_ = 7;
};

#endif // MAP_H