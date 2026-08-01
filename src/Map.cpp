#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"

Map::Map() {}

void Map::AddKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
}

void Map::AddMapPoint(MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);
}

void Map::EraseKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
}

std::vector<KeyFrame*> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
}

std::vector<MapPoint*> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
}

long unsigned int Map::GetMapPointsInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::GetKeyFramesInMap()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

void Map::Clear()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mspMapPoints.clear();
    mspKeyFrames.clear();
    mvpReferenceMapPoints.clear();
}