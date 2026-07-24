#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <memory>
#include <vector>
#include <list>
#include <mutex>
#include <Eigen/Core>

class Frame;

class MapPoint
{
public:
    typedef std::shared_ptr<MapPoint> Ptr;

    MapPoint() = default;
    MapPoint(unsigned long id, const Eigen::Vector3d &pos);

    unsigned long id_ = 0;    // 地图点ID
    bool is_outlier_ = false; // 是否为离群点

    Eigen::Vector3d getPos();
    void setPos(const Eigen::Vector3d &pos);

    bool isBad();
    void setBad(bool is_bad);

    void addObservation(std::shared_ptr<Frame> frame);
    void removeObservation(std::shared_ptr<Frame> frame);
    std::list<std::weak_ptr<Frame>> getObservations();

    static MapPoint::Ptr createMapPoint(const Eigen::Vector3d &pos_world);

private:
    static unsigned long id_counter_;
    Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();

    std::mutex data_mutex_; // 保护 pos_ 和 is_outlier_
    std::mutex obs_mutex_;  // 保护 observations_
    static std::mutex id_mutex_;
    std::list<std::weak_ptr<Frame>> observations_;
};

#endif // MAPPOINT_H