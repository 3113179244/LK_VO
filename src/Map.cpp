#include "Map.h"

void Map::insertKeyFrame(Frame::Ptr frame)
{
    std::unique_lock<std::mutex> lock(data_mutex_);

    keyframes_[frame->id()] = frame;
    active_keyframes_[frame->id()] = frame;

    // 如果局部关键帧数量超过上限，滑出最老的关键帧
    if (active_keyframes_.size() > num_active_keyframes_)
    {
        // 找到 ID 最小（最老）的关键帧并移除
        unsigned long min_id = std::numeric_limits<unsigned long>::max();
        for (auto &kf : active_keyframes_)
        {
            if (kf.first < min_id)
            {
                min_id = kf.first;
            }
        }
        active_keyframes_.erase(min_id);
    }
}

void Map::insertMapPoint(MapPoint::Ptr map_point)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    landmarks_[map_point->id_] = map_point;
    active_landmarks_[map_point->id_] = map_point;
}

Map::LandmarksType Map::getAllMapPoints()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return landmarks_;
}

Map::KeyframesType Map::getAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return keyframes_;
}

Map::LandmarksType Map::getActiveMapPoints()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return active_landmarks_;
}

Map::KeyframesType Map::getActiveKeyFrames()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return active_keyframes_;
}

void Map::removeMapPoint(MapPoint::Ptr map_point)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    landmarks_.erase(map_point->id_);
    active_landmarks_.erase(map_point->id_);
}

void Map::cleanMap()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    int cnt_landmark_removed = 0;
    for (auto iter = active_landmarks_.begin(); iter != active_landmarks_.end();)
    {
        if (iter->second->is_outlier_)
        {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
        }
        else
        {
            ++iter;
        }
    }
}