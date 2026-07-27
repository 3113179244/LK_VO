#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include <vector>
#include <memory>
#include <mutex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Camera;

// 种子点：记录需要估计深度的像素点及其概率分布
struct Seed
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int id;               // 唯一标识 ID
    Eigen::Vector2f px;   // 种子点在参考关键帧上的像素坐标
    Eigen::Vector3f f;    // 归一化相机坐标系下的单位方向向量

    float a;              // Beta 分布参数 a (内点频数)
    float b;              // Beta 分布参数 b (外点频数)
    float mu;             // 逆深度高斯分布均值 (1/d)
    float z_range;        // 深度范围阈值/不确定度
    float sigma2;         // 逆深度高斯分布方差

    bool is_converged;    // 是否收敛

    Seed(int seed_id, const Eigen::Vector2f &pixel, const Eigen::Vector3f &ray, float depth_init, float depth_min)
        : id(seed_id), px(pixel), f(ray.normalized()), a(10.0f), b(10.0f), is_converged(false)
    {
        // 逆深度初始化
        mu = 1.0f / depth_init;
        float max_depth_inv = 1.0f / depth_min;
        sigma2 = (max_depth_inv * max_depth_inv) / 36.0f; // 3-sigma 覆盖区间
        z_range = depth_init;
    }
};

class DepthFilter
{
public:
    typedef std::shared_ptr<DepthFilter> Ptr;

    DepthFilter(std::shared_ptr<Camera> pCamera);
    ~DepthFilter() = default;

    /**
     * @brief 添加新的 Seed（种子点）
     */
    void AddSeed(const Eigen::Vector2f &pt, float init_depth = 2.0f, float min_depth = 0.5f);

    /**
     * @brief 传入新图像与当前位姿，更新所有未收敛的种子点
     * @param cur_img 当前帧图像
     * @param T_cur_ref 参考帧到当前帧的位姿变换矩阵 T_cur_ref
     */
    void Update(const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref);

    /**
     * @brief 获取所有已收敛的 Seed，转换为 3D 点后可从 filter 中清除
     */
    std::vector<Seed> GetConvergedSeeds();

private:
    /**
     * @brief 极线搜索与 NCC 匹配
     */
    bool EpipolarSearch(const Seed &seed, const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref,
                        Eigen::Vector2d &best_curr_px, float &depth_best, float &variance_best);

    /**
     * @brief 贝叶斯概率更新（高斯-均匀混合模型 / Vogiatzis 算法）
     */
    void UpdateSeed(Seed &seed, float x, float tau_sq);

    /**
     * @brief 计算图像上两点间的 NCC 相似度
     */
    float ComputeNCC(const cv::Mat &ref_img, const cv::Mat &cur_img, const Eigen::Vector2f &pt_ref, const Eigen::Vector2f &pt_cur);

private:
    std::shared_ptr<Camera> mpCamera;
    std::vector<Seed> mvSeeds;
    cv::Mat mRefImg; // 参考帧图像
    int mNextSeedId;
    std::mutex mMutexSeeds;
};

#endif // DEPTH_FILTER_H