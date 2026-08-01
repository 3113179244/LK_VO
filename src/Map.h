#ifndef MAP_H
#define MAP_H

#include <set>
#include <vector>
#include <mutex>

class KeyFrame;
class MapPoint;

class Map
{
public:
    Map();

    // ---- 关键帧与地图点管理 ----
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void EraseKeyFrame(KeyFrame* pKF);

    // ---- 获取全局数据副本 (用于 Visualization / Loop Closing) ----
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();

    long unsigned int GetMapPointsInMap();
    long unsigned int GetKeyFramesInMap();

    void Clear();

private:
    std::set<MapPoint*> mspMapPoints;
    std::set<KeyFrame*> mspKeyFrames;

    std::vector<MapPoint*> mvpReferenceMapPoints;

    std::mutex mMutexMap;
};

#endif // MAP_H