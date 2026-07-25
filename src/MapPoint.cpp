#include "MapPoint.h"
#include "Frame.h"

MapPoint::MapPoint(const long unsigned int id, const cv::Mat &position)
    : mId(id), mWorldPos(position.clone()), mnObs(0), mbBad(false) {}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

void MapPoint::AddObservation(std::shared_ptr<Frame> pFrame, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pFrame))
        return;
    mObservations[pFrame] = idx;
    mnObs++;
}

void MapPoint::RemoveObservation(std::shared_ptr<Frame> pFrame)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pFrame))
    {
        mObservations.erase(pFrame);
        mnObs--;
    }
}

std::map<std::shared_ptr<Frame>, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::GetObservedCount()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mnObs;
}

void MapPoint::SetBadFlag()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mbBad = true;
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mbBad;
}