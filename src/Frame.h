#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>

// 前向声明
class Camera;
class MapPoint;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double timestamp,
          std::shared_ptr<Camera> camera, const int id);

    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();
    Eigen::Vector3f GetCameraCenter();

    bool isInFrustum(const MapPoint *pMapPoint, float viewingCosLimit);

    long unsigned int mId;
    double mTimeStamp;
    std::shared_ptr<Camera> mpCamera;
    float mbf;
    Eigen::Matrix4f mTcw;
    Eigen::Matrix3f mRcw;
    Eigen::Vector3f mtcw;
    Eigen::Matrix3f mRwc;
    Eigen::Vector3f mOw;

    int N;
    std::vector<cv::KeyPoint> mvKeys;
    std::vector<cv::KeyPoint> mvKeysRight;

    std::vector<float> mvuRight;
    std::vector<float> mvDepth;
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
    std::vector<bool> mvbOutlier;
    
    std::mutex mMutexPose;
};

#endif // FRAME_H