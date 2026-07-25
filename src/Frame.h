#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <mutex>

class Camera;   // 相机内参模型
class MapPoint; // 3D地图点

class Frame
{
public:
    // 智能指针定义，方便内存管理
    typedef std::shared_ptr<Frame> Ptr;

    // 构造函数：传入左右图像、时间戳、相机模型等
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double timestamp,
          std::shared_ptr<Camera> camera, const int id);

    // 设置/获取当前帧的位姿 T_cw (世界坐标系到相机坐标系的变换矩阵)
    void SetPose(const Eigen::Matrix4f &Tcw);
    Eigen::Matrix4f GetPose();

    // 获取相机光心在世界坐标系下的位置
    Eigen::Vector3f GetCameraCenter();

    // 判断某个特征点是否在视野内
    bool isInFrustum(const MapPoint *pMP, float viewingCosLimit);

public:
    long unsigned int mId; // 帧的唯一ID
    double mTimeStamp;     // 时间戳
    std::shared_ptr<Camera> mpCamera; // 相机模型（包含焦距 fx, fy, cx, cy, 畸变参数）
    float mbf;                        // baseline * fx (基线乘以焦距，用于计算深度)
    Eigen::Matrix4f mTcw;
    Eigen::Matrix3f mRcw; // 旋转矩阵 R
    Eigen::Vector3f mtcw; // 平移向量 t
    Eigen::Matrix3f mRwc; // R^T
    Eigen::Vector3f mOw;  // 光心位置: -R^T * t

    int N;                                 // 提取到的特征点总数量 (以左目为准)
    std::vector<cv::KeyPoint> mvKeys;      // 左目特征点 (去畸变后的坐标)
    std::vector<cv::KeyPoint> mvKeysRight; // 右目特征点 (仅在匹配时临时使用，可优化掉)

    // 长度均等于 N (左目特征点数)
    std::vector<float> mvuRight; // 左目特征点在右目图像中匹配到的横坐标 u_right (如果未匹配上则为 -1)
    std::vector<float> mvDepth;  // 每个特征点对应的深度 z。 z = mbf / (u_left - u_right)
    // 长度等于 N，记录每个特征点关联的 3D 地图点 (未关联则为 nullptr)
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
    // 记录特征点是否为外点 (Outlier)，在优化时标记
    std::vector<bool> mvbOutlier;

private:
    std::mutex mMutexPose; // 保护位姿更新的互斥锁
};

#endif // FRAME_H