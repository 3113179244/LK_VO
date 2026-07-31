#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>
#include <mutex>
#include <thread>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

// 前向声明
class Map;
class Frame;
class MapPoint;

class Viewer {
public:
    Viewer(std::shared_ptr<Map> pMap);
    ~Viewer() = default;
    
    void Run();
    void RequestFinish();
    void RequestStop();
    void Release();
    bool isFinished();
    bool isStopped();

    // 设置当前帧位姿（由System调用）
    void SetCurrentCameraPose(const Eigen::Matrix4d& Tcw);
    // 设置当前帧图像（用于2D显示）
    void SetCurrentFrameImage(const cv::Mat& img);
    // 重置请求（由外部检查）
    void RequestReset();
    bool CheckReset();

private:
    bool Stop(); // 内部检查停止
    bool CheckFinish();
    void SetFinish();
    
    void DrawMapPoints();
    void DrawKeyFrames();
    void DrawGraph();
    void DrawCurrentCamera();
    void DrawCamera(const Eigen::Matrix4f& Tcw, float r, float g, float b, float scale = 0.1f);

    // 【新增】获取当前相机的 Pangolin OpenGL 矩阵（支持跟随镜头）
    pangolin::OpenGlMatrix GetCurrentOpenGLCameraMatrix();

    std::shared_ptr<Map> mpMap;
    
    // 线程同步
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    bool mbFinished;
    
    std::mutex mMutexStop;
    bool mbStopRequested;
    bool mbStopped;

    // 当前相机位姿
    std::mutex mMutexCurrentCam;
    Eigen::Matrix4d mCurrentTcw;

    // 当前帧图像
    std::mutex mMutexImg;
    cv::Mat mCurrentFrameImg;

    // 显示选项（内部状态）
    bool mbFollowCamera;
    bool mbShowPoints;
    bool mbShowKeyFrames;
    bool mbShowGraph;

    // 可视化参数
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    // 重置请求标志
    std::mutex mMutexReset;
    bool mbResetRequested;
};

#endif // VIEWER_H