#ifndef VIEWER_H
#define VIEWER_H

#include <pangolin/pangolin.h>
#include <mutex>
#include <thread>
#include <memory>

// 前向声明
class Map;
class Frame;

class Viewer {
public:
    Viewer(std::shared_ptr<Map> pMap);
    
    void Run();
    void RequestFinish();
    bool isFinished();

private:
    void DrawMapPoints();
    void DrawKeyFrames();
    void DrawCurrentCamera();

    pangolin::OpenGlMatrix GetCurrentOpenGLCameraMatrix();

private:
    std::shared_ptr<Map> mpMap;
    
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    bool mbFinished;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
};

#endif // VIEWER_H