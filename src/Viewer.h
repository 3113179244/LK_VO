#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>
#include <mutex>
#include <thread>
#include <memory>
#include <Eigen/Core>

// 前向声明
class Map;
class Frame;
class MapPoint;

class Viewer {
public:
    Viewer(std::shared_ptr<Map> pMap);
    
    void Run();
    void RequestFinish();
    bool isFinished();

    // 设置当前帧位姿（由System调用）
    void SetCurrentCameraPose(const Eigen::Matrix4d& Tcw);

private:
    void DrawMapPoints();
    void DrawKeyFrames();
    void DrawCurrentCamera();

    // 辅助函数：绘制一个相机模型（颜色RGB）
    void DrawCamera(const Eigen::Matrix4f& Tcw, float r, float g, float b, float scale = 0.1f);

    std::shared_ptr<Map> mpMap;
    
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    bool mbFinished;

    // 当前相机位姿（由外部更新）
    std::mutex mMutexCurrentCam;
    Eigen::Matrix4d mCurrentTcw;

    // 可视化参数
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
};

#endif // VIEWER_H