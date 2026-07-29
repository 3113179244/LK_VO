#ifndef MAP_H
#define MAP_H

#include <memory>
#include <vector>
#include <map>
#include <mutex>

class Frame;
class MapPoint;

class Map {
public:
    typedef std::shared_ptr<Map> Ptr;
    Map() {}
    ~Map() = default;

    void InsertKeyFrame(std::shared_ptr<Frame> frame);
    void InsertMapPoint(std::shared_ptr<MapPoint> map_point);
    void EraseMapPoint(std::shared_ptr<MapPoint> map_point);
    void EraseKeyFrame(unsigned long id); 
    std::vector<std::shared_ptr<Frame>> GetAllKeyFrames();
    std::vector<std::shared_ptr<MapPoint>> GetAllMapPoints();
    int GetKeyFramesInMap();
    int GetMapPointsInMap();

private:
    std::mutex mMutexMap;
    std::map<unsigned long, std::shared_ptr<Frame>> mKeyFrames;
    std::map<unsigned long, std::shared_ptr<MapPoint>> mMapPoints;
};

#endif