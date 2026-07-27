#ifndef MAP_H
#define MAP_H

#include <memory>
#include <vector>
#include <map>
#include <mutex>

// 前向声明
class Frame;
class MapPoint;

class Map {
public:
    typedef std::shared_ptr<Map> Ptr;

    Map();
    ~Map() = default;

    void InsertKeyFrame(std::shared_ptr<Frame> frame);
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);
    void EraseMapPoint(std::shared_ptr<MapPoint> map_point);
    void EraseOldestKeyFrame();

    std::vector<std::shared_ptr<Frame>> GetAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    std::vector<std::shared_ptr<Frame>> GetActiveKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetActiveMapPoints();

    int GetKeyFramesInMap();
    int GetMapPointsInMap();
    void SetNumActiveKeyFrames(int num);

private:
    std::mutex mMutexMap;
    std::map<unsigned long, std::shared_ptr<Frame>> mKeyFrames;
    std::map<unsigned long, std::shared_ptr<MapPoint>> mMapPoints;
    std::map<unsigned long, std::shared_ptr<Frame>> mActiveKeyFrames;
    std::map<unsigned long, std::shared_ptr<MapPoint>> mActiveMapPoints;
    int mNumActiveKeyframes;
};

#endif // MAP_H