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
class FeatureDetector;

class Frame
{
public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame(const cv::Mat &image0, const cv::Mat &image1, const double dtimestamp, const int FrameId);

    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();
    Eigen::Vector3f GetCameraCenter();

    bool isInFrustum(const MapPoint *pMapPoint, float viewingCosLimit);

    unsigned long int mFrameId; // 帧图像的全局唯一身份证号
    double mTimeStamp;
    int iFeaturePointnums; // 当前帧的特征点数量
    std::vector<int> mvFeatureIds;//特征点的ID
    std::vector<cv::KeyPoint> mvleftpixel;               // 第 i 个 2D 像素点
    std::vector<cv::KeyPoint> mvrightpixel;              // 第 i 个 2D 像素点
    std::vector<int> mvTrackCnt;                         // 特征点被连续追踪的次数
    std::vector<float> mvInverseDepth;                   // 第 i 个点的逆深度
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints; // 第 i 个点对应的 3D 地图点
    std::vector<bool> mvbOutlier;                        // 误匹配/坏点标记列表

private:
    Eigen::Matrix4f mTcw; // 世界坐标系到当前相机坐标系的旋转平移矩阵
    Eigen::Matrix3f mRcw; // 世界坐标系到当前相机坐标系的旋转矩阵
    Eigen::Vector3f mtcw; // 世界坐标系到当前相机坐标系的平移向量
    Eigen::Matrix3f mRwc; // 相机坐标系到世界坐标系的旋转矩阵
    Eigen::Vector3f mOw;  // 相机光心在世界坐标系下的3D物理坐标。
    std::mutex mMutexPose;
};

#endif // FRAME_H