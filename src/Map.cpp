#include "Map.h"
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
Map::Map() : mNumActiveKeyframes(10) {}

void Map::InsertKeyFrame(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mKeyFrames[frame->mId] = frame;
    mActiveKeyFrames[frame->mId] = frame;
}

void Map::InsertMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mMapPoints[map_point->mId] = map_point;
    mActiveMapPoints[map_point->mId] = map_point;
}

void Map::EraseMapPoint(std::shared_ptr<MapPoint> map_point)
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    mMapPoints.erase(map_point->mId);
    mActiveMapPoints.erase(map_point->mId);
    map_point->SetBadFlag();
}

void Map::EraseOldestKeyFrame()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    if (mActiveKeyFrames.size() > (size_t)mNumActiveKeyframes)
    {
        // 查找时间戳最老/ID最小的一帧
        unsigned long min_id = std::numeric_limits<unsigned long>::max();
        for (auto &kf : mActiveKeyFrames)
        {
            if (kf.first < min_id)
            {
                min_id = kf.first;
            }
        }
        mActiveKeyFrames.erase(min_id);
    }
}

std::vector<std::shared_ptr<Frame>> Map::GetAllKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<Frame>> res;
    for (auto &kf : mKeyFrames)
        res.push_back(kf.second);
    return res;
}

std::vector<std::shared_ptr<MapPoint>> Map::GetAllMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<MapPoint>> res;
    for (auto &mp : mMapPoints)
        res.push_back(mp.second);
    return res;
}

std::vector<std::shared_ptr<Frame>> Map::GetActiveKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<Frame>> res;
    for (auto &kf : mActiveKeyFrames)
        res.push_back(kf.second);
    return res;
}

std::vector<std::shared_ptr<MapPoint>> Map::GetActiveMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexMap);
    std::vector<std::shared_ptr<MapPoint>> res;
    for (auto &mp : mActiveMapPoints)
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

void Map::SetNumActiveKeyFrames(int num)
{
    mNumActiveKeyframes = num;
}