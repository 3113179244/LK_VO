#ifndef _DEPTH_FILTER_H
#define _DEPTH_FILTER_H

#include <vector>
#include <memory>
#include <mutex>
#include <cmath>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

class Camera;

struct Seed
{
    int id;                     // 对应左目特征点索引
    Eigen::Vector2f px;         // 左目像素坐标
    Eigen::Vector3f f;          // 归一化相机射线（左目）

    // Beta 分布参数（高斯概率）
    float a, b;
    // 逆深度均值和方差
    float mu, sigma2;
    // 深度搜索范围（用于均匀分布部分）
    float z_range;

    bool is_converged;

    Seed(int _id, const Eigen::Vector2f &_px, const Eigen::Vector3f &_ray,
               float init_depth, float min_depth)
        : id(_id), px(_px), f(_ray.normalized()), a(10.0f), b(10.0f),
          is_converged(false)
    {
        mu = 1.0f / init_depth;
        float max_depth_inv = 1.0f / min_depth;
        sigma2 = (max_depth_inv * max_depth_inv) / 36.0f; // 6-sigma 覆盖
        z_range = init_depth;
    }
};

class DepthFilter
{
public:
    typedef std::shared_ptr<DepthFilter> Ptr;

    DepthFilter(std::shared_ptr<Camera> pCamera, float baseline, float max_disparity = 100.0f);

    // 添加新种子（对应左目特征点）
    void AddSeed(const Eigen::Vector2f &pt_left, float init_depth = 2.0f, float min_depth = 0.5f);

    // 用双目图像对更新所有活跃种子
    void Update(const cv::Mat &left_img, const cv::Mat &right_img);

    // 提取已收敛的种子（并移出活跃列表）
    std::vector<Seed> GetConvergedSeeds();

private:
    // 对单个种子进行双目匹配，返回观测到的逆深度及方差
    bool ComputeObservation(const Seed &seed,
                                  const cv::Mat &left_img,
                                  const cv::Mat &right_img,
                                  float &tau_obs, float &sigma2_obs);

    // 贝叶斯更新（与单目相同）
    void UpdateSeed(Seed &seed, float tau_obs, float sigma2_obs);

    // NCC 计算（7x7 窗口）
    float ComputeNCC(const cv::Mat &patch_ref, const cv::Mat &patch_cur);

private:
    std::shared_ptr<Camera> mpCamera;
    float mBaseline;          // 双目基线（米）
    float mMaxDisparity;      // 最大搜索视差（像素）
    float mFocalLength;       // 焦距（假设 fx = fy）
    int mNextSeedId;

    std::vector<Seed> mvSeeds;
    std::mutex mMutexSeeds;

    const int mPatchSize = 7; // 匹配窗口半宽
};

#endif // _DEPTH_FILTER_H