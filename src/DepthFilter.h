#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include <vector>
#include <memory>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// 前向声明
class Camera;

struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id;
    Eigen::Vector2f px;
    Eigen::Vector3f f;

    float a;
    float b;
    float mu;
    float z_range;
    float sigma2;

    bool is_converged;

    Seed(int seed_id, const Eigen::Vector2f &pixel, const Eigen::Vector3f &ray, float depth_init, float depth_min)
        : id(seed_id), px(pixel), f(ray.normalized()), a(10.0f), b(10.0f), is_converged(false)
    {
        mu = 1.0f / depth_init;
        float max_depth_inv = 1.0f / depth_min;
        sigma2 = (max_depth_inv * max_depth_inv) / 36.0f;
        z_range = depth_init;
    }
};

class DepthFilter
{
public:
    typedef std::shared_ptr<DepthFilter> Ptr;

    DepthFilter(std::shared_ptr<Camera> pCamera);
    ~DepthFilter() = default;

    void AddSeed(const Eigen::Vector2f &pt, float init_depth = 2.0f, float min_depth = 0.5f);
    void Update(const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref);
    std::vector<Seed> GetConvergedSeeds();

private:
    bool EpipolarSearch(const Seed &seed, const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref,
                        Eigen::Vector2d &best_curr_px, float &depth_best, float &variance_best);
    void UpdateSeed(Seed &seed, float x, float tau_sq);
    float ComputeNCC(const cv::Mat &ref_img, const cv::Mat &cur_img, const Eigen::Vector2f &pt_ref, const Eigen::Vector2f &pt_cur);

private:
    std::shared_ptr<Camera> mpCamera;
    std::vector<Seed> mvSeeds;
    cv::Mat mRefImg;
    int mNextSeedId;
    std::mutex mMutexSeeds;
};

#endif // DEPTH_FILTER_H