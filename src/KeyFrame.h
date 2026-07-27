#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>
#include <map>

// 前向声明
class Frame;
class Camera;
class Map;
class MapPoint;

class KeyFrame
{
public:
    typedef std::shared_ptr<KeyFrame> Ptr;

    KeyFrame(std::shared_ptr<Frame> pFrame);

    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();
    Eigen::Vector3f GetCameraCenter();

    std::vector<std::shared_ptr<MapPoint>> GetMapPoints();
    std::shared_ptr<MapPoint> GetMapPoint(const size_t idx);

    void AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t idx);
    void EraseMapPointMatch(const size_t idx);
    void EraseMapPointMatch(std::shared_ptr<MapPoint> pMP);

    void AddConnection(std::shared_ptr<KeyFrame> pKF, const int weight);
    void EraseConnection(std::shared_ptr<KeyFrame> pKF);
    std::map<std::shared_ptr<KeyFrame>, int> GetConnectedKeyFrames();

public:
    const long unsigned int mId;
    const long unsigned int mFrameId;
    const double mTimeStamp;

    std::shared_ptr<Camera> mpCamera;
    const float mbf;
    const int N;

    std::vector<cv::KeyPoint> mvKeysLeft;
    std::vector<int> mvFeatureIds;

private:
    Eigen::Matrix4f mTcw;
    Eigen::Vector3f mOw;

    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
    std::map<std::shared_ptr<KeyFrame>, int> mConnectedKeyFrameWeights;

    std::mutex mMutexPose;
    std::mutex mMutexFeatures;
    std::mutex mMutexConnections;
};

#endif // KEYFRAME_H