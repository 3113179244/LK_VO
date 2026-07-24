#include "MapPoint.h"
#include "Frame.h"

unsigned long MapPoint::id_counter_ = 0;

MapPoint::MapPoint(unsigned long id, const Eigen::Vector3d &pos)
    : id_(id), pos_(pos) {}

MapPoint::Ptr MapPoint::createMapPoint(const Eigen::Vector3d &pos)
{
    std::unique_lock<std::mutex> lock(id_mutex_);
    return std::make_shared<MapPoint>(id_counter_++, pos);
}

Eigen::Vector3d MapPoint::getPos()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return pos_;
}

void MapPoint::setPos(const Eigen::Vector3d &pos)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    pos_ = pos;
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    return is_outlier_;
}

void MapPoint::setBad(bool is_bad)
{
    std::unique_lock<std::mutex> lock(data_mutex_);
    is_outlier_ = is_bad;
}

void MapPoint::addObservation(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(obs_mutex_);
    observations_.push_back(frame);
}

void MapPoint::removeObservation(std::shared_ptr<Frame> frame)
{
    std::unique_lock<std::mutex> lock(obs_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end(); ++iter)
    {
        if (iter->lock() == frame)
        {
            observations_.erase(iter);
            break;
        }
    }
}

std::list<std::weak_ptr<Frame>> MapPoint::getObservations()
{
    std::unique_lock<std::mutex> lock(obs_mutex_);
    return observations_;
}