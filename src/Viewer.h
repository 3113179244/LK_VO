#ifndef VIEWER_H
#define VIEWER_H

#include <GL/glew.h>
#include <pangolin/pangolin.h>
#include <mutex>
#include <thread>
#include <memory>
#include "Map.h"
#include "Frame.h"

class Viewer {
public:
    // 传入全局地图的指针，用于读取需要渲染的数据
    Viewer(std::shared_ptr<Map> pMap);
    
    // 主循环函数，将被放入单独的线程中执行
    void Run();

    // 允许外部（如 Tracker）请求终止可视化线程
    void RequestFinish();
    bool isFinished();

private:
    // 绘制全局地图点和局部活跃地图点
    void DrawMapPoints();
    
    // 绘制关键帧相机位姿的视锥体
    void DrawKeyFrames();
    
    // 绘制当前相机的位姿
    void DrawCurrentCamera();

    // 将 Eigen::Matrix4f (Tcw) 转换为 Pangolin 支持的 OpenGL 矩阵格式
    pangolin::OpenGlMatrix GetCurrentOpenGLCameraMatrix();

private:
    std::shared_ptr<Map> mpMap;
    
    // 线程控制标志位与锁
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    bool mbFinished;

    // 可视化参数设置
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
};

#endif // VIEWER_H