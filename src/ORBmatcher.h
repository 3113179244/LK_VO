#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class Frame;
class KeyFrame;
class MapPoint;

class ORBmatcher
{
public:
    // 构造函数：指定最佳距离与次佳距离的比值阈值 (nnratio)，通常取 0.6 ~ 0.8
    ORBmatcher(float nnratio = 0.7f, bool checkOri = true);
    ~ORBmatcher() = default;

    // ---- 1. 汉明距离计算基础接口 ----
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // ---- 2. 追踪阶段：地图点投影到 Frame 匹配 ----
    // 将 MapPoints 投影到当前 Frame，寻找最佳匹配（用于 Tracking / TrackLocalMap）
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th = 3.0f);

    // ---- 3. 帧间匹配：基于上一帧的位姿投影搜索 ----
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th);

    // ---- 4. 关键帧建图：寻找三角化候选点 (极线约束) ----
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                               std::vector<std::pair<size_t, size_t>> &vMatchedPairs);

    // ---- 5. 地图点融合：通过投影融合重复的 MapPoint ----
    int Fuse(KeyFrame* pKF, const std::vector<MapPoint *> &vpMapPoints, const float th = 3.0f);

public:
    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;

protected:
    // 方向一致性校验 (主方向直方图剔除错配)
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &idx1, int &idx2, int &idx3);

    float mfNNratio;
    bool mbCheckOrientation;
};

#endif // ORBMATCHER_H