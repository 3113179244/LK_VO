#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

class MapPoint;
class ORBextractor;
class ORBVocabulary;

#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class Frame
{
public:
    Frame();

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, 
          ORBextractor* extractorLeft, ORBextractor* extractorRight, 
          ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    void ExtractORB(int flag, const cv::Mat &im);

    // 设置相机位姿 (使用 Eigen)
    void SetPose(const Eigen::Matrix4f &Tcw);

    // 从相机位姿更新内部变换矩阵
    void UpdatePoseMatrices();

    // 获取相机光心在世界坐标系下的位置
    inline Eigen::Vector3f GetCameraCenter() { return mOw; }

    // 获取相机旋转矩阵的逆 (相机到世界)
    inline Eigen::Matrix3f GetRotationInverse() { return mRwc; }

    // 将带有深度信息的双目特征点反投影为 3D 世界坐标
    Eigen::Vector3f UnprojectStereo(const int &i);

    std::vector<size_t> GetFeaturesInArea(const float &x, const float &y, const float &r, 
                                          const int minLevel = -1, const int maxLevel = -1) const;

    static long unsigned int nNextId;
    long unsigned int mnId;
    double mTimeStamp;

    cv::Mat mK;
    static float fx, fy, cx, cy, invfx, invfy;
    cv::Mat mDistCoef;
    float mbf;
    float mb;
    float mThDepth;

    int N;
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight, mvKeysUn;
    cv::Mat mDescriptors, mDescriptorsRight;
    
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    std::vector<MapPoint*> mvpMapPoints;
    std::vector<bool> mvbOutlier;

    // ---- 相机位姿 (Eigen) ----
    Eigen::Matrix4f mTcw;

    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    static float mnMinX, mnMaxX, mnMinY, mnMaxY;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    ORBextractor* mpORBextractorLeft;
    ORBextractor* mpORBextractorRight;
    ORBVocabulary* mpORBvocabulary;

    static bool mbInitialComputations;

private:
    void ComputeStereoMatches();
    void ComputeImageBounds(const cv::Mat &imLeft);
    void AssignFeaturesToGrid();
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    // 位姿内部矩阵 (Eigen)
    Eigen::Matrix3f mRcw;
    Eigen::Vector3f mtcw;
    Eigen::Matrix3f mRwc;
    Eigen::Vector3f mOw;
};

#endif // FRAME_H