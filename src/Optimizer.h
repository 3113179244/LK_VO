#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include "Frame.h"
#include "Camera.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"

class Optimizer {
public:
    /**
     * @brief 前端位姿优化 (Motion-only BA)
     * 仅优化当前帧的位姿 Tcw，地图点固定
     * @param pFrame 当前帧
     * @return 成功参与优化的内点 (Inliers) 数量
     */
    static int PoseOptimization(std::shared_ptr<Frame> pFrame);

    /**
     * @brief 后端局部集束调整 (Local BA)
     * 优化局部窗口内的关键帧位姿和对应的地图点 3D 坐标
     * @param pMap 全局地图 (内部会提取 Active KeyFrames 和 Active MapPoints)
     */
    static void LocalBundleAdjustment(std::shared_ptr<Map> pMap);

    /**
     * @brief 全局集束调整 (Global BA) - 通常用于回环检测后或系统结束时
     * 优化地图中所有的关键帧和地图点
     * @param pMap 全局地图
     */
    static void GlobalBundleAdjustment(std::shared_ptr<Map> pMap);
};

#endif // OPTIMIZER_H