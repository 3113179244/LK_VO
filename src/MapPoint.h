#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/opencv.hpp>
#include <map>
#include <memory>
#include <mutex>

// 前向声明
class Frame;

class MapPoint {
public:
    typedef std::shared_ptr<MapPoint> Ptr;

    MapPoint(const long unsigned int id, const cv::Mat& position);

    void SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    void AddObservation(std::shared_ptr<Frame> pFrame, size_t idx);
    void RemoveObservation(std::shared_ptr<Frame> pFrame);
    
    std::map<std::shared_ptr<Frame>, size_t> GetObservations();
    int GetObservedCount();

    void SetBadFlag();
    bool isBad();

public:
    const long unsigned int mId;

private:
    cv::Mat mWorldPos;
    std::map<std::shared_ptr<Frame>, size_t> mObservations; 
    int mnObs;
    bool mbBad;

    std::mutex mMutexPos;
    std::mutex mMutexFeatures;
};

#endif // MAPPOINT_H