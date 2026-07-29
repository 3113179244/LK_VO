#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"

void Map::InsertKeyFrame(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mKeyFrames[frame->mFrameId] = frame;
}

void Map::InsertMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mMapPoints[map_point->mId] = map_point;
}

void Map::EraseMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mMapPoints.erase(map_point->mId);
    map_point->SetBadFlag();
}

void Map::EraseKeyFrame(unsigned long id)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mKeyFrames.erase(id);
}

std::vector<std::shared_ptr<Frame>> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<Frame>> res;
    res.reserve(mKeyFrames.size());
    for (auto &kf : mKeyFrames)
        res.push_back(kf.second);
    return res;
}

std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<MapPoint>> res;
    res.reserve(mMapPoints.size());
    for (auto &mp : mMapPoints)
        res.push_back(mp.second);
    return res;
}

int Map::GetKeyFramesInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mKeyFrames.size();
}

int Map::GetMapPointsInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mMapPoints.size();
}