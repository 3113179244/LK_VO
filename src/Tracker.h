#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>

#include "Config.h"
#include "Camera.h"
#include "Frame.h"
#include "FeatureDetector.h"
#include "Map.h"
#include "Optimizer.h"

// 定义追踪器的状态机
enum eTrackingState {
    SYSTEM_NOT_READY = -1,
    NO_IMAGES_YET = 0,
    NOT_INITIALIZED = 1,
    OK = 2,
    LOST = 3
};

class Tracker {
public:
    Tracker(std::shared_ptr<Camera> pCamera, std::shared_ptr<Map> pMap);
    ~Tracker() = default;

    // 系统的主入口：传入当前时间戳和左右目图像，返回当前位姿 Tcw
    Eigen::Matrix4d GrabImageStereo(const cv::Mat& imLeft, const cv::Mat& imRight, const double timestamp);

    // 获取当前追踪状态
    eTrackingState GetTrackingState() const { return mState; }

private:
    // --- 核心处理步骤 ---
    
    // 1. 双目初始化：第一帧时建立初始地图
    bool StereoInitialization();

    // 2. 帧间追踪：利用光流追踪的结果，预测当前帧位姿并进行 PnP 求解
    bool TrackPreviousFrame();

    // 3. 局部地图追踪：与局部地图中的 MapPoints 匹配，进一步优化位姿
    bool TrackLocalMap();

    // 4. 关键帧决策：判断是否需要插入新关键帧
    bool NeedNewKeyFrame();
    
    void CreateNewKeyFrame();

private:
    // 状态机
    eTrackingState mState;

    // 核心组件指针
    std::shared_ptr<Camera> mpCamera;
    std::shared_ptr<Map> mpMap;
    
    // 特征追踪器 (负责光流和双目极线搜索)
    FeatureDetector mFeatureTracker;

    // 帧管理
    std::shared_ptr<Frame> mCurrentFrame;
    std::shared_ptr<Frame> mLastFrame;

    // 位姿与运动模型
    Eigen::Matrix4d mVelocity;// 恒速运动模型，用于位姿预测的 T_curr_prev

    // 配置参数
    int mMaxFrames;
};

#endif // TRACKER_H