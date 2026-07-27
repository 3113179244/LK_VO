#include "DepthFilter.h"
#include "Camera.h"
#include <cmath>
#include <algorithm>

DepthFilter::DepthFilter(std::shared_ptr<Camera> pCamera)
    : mpCamera(pCamera), mNextSeedId(0) {}

void DepthFilter::AddSeed(const Eigen::Vector2f &pt, float init_depth, float min_depth)
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);
    // 将求出的 Vector3d 显式转换为 Vector3f 供 Seed 结构体使用
    Eigen::Vector3f f = mpCamera->Pixel2Camera(pt.cast<double>()).cast<float>();
    mvSeeds.emplace_back(mNextSeedId++, pt, f, init_depth, min_depth);
}

void DepthFilter::Update(const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref)
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);

    for (auto &seed : mvSeeds)
    {
        if (seed.is_converged)
            continue;

        Eigen::Vector2d best_px = Eigen::Vector2d::Zero();
        float tau_inv = 0.0f;
        float sigma2_tau = 0.0f;

        // 1. 沿极线搜索并提取观测到的逆深度 tau_inv 以及观测方差 sigma2_tau
        if (EpipolarSearch(seed, cur_img, T_cur_ref, best_px, tau_inv, sigma2_tau))
        {
            // 2. 贝叶斯滤波器更新
            UpdateSeed(seed, tau_inv, sigma2_tau);

            // 3. 收敛条件判定：方差小于给定阈值或者标准差占逆深度的比例极小
            if (std::sqrt(seed.sigma2) < 0.005f * seed.mu || seed.a / (seed.a + seed.b) > 0.85f)
            {
                seed.is_converged = true;
            }
        }
    }
}

// 注意：加上 DepthFilter:: 类作用域修饰符
bool DepthFilter::EpipolarSearch(const Seed &seed, const cv::Mat &cur_img, const Eigen::Matrix4f &T_cur_ref,
                                 Eigen::Vector2d &best_curr_px, float &depth_best, float &variance_best)
{
    Eigen::Matrix3f R_cur_ref = T_cur_ref.block<3, 3>(0, 0);
    Eigen::Vector3f t_cur_ref = T_cur_ref.block<3, 1>(0, 3);

    // 计算极线端点 (对应 3-sigma 深度区间)
    float d_min = 1.0f / (seed.mu + 3.0f * std::sqrt(seed.sigma2));
    float d_max = 1.0f / std::max(0.00001f, seed.mu - 3.0f * std::sqrt(seed.sigma2));

    Eigen::Vector3f P_min = R_cur_ref * (seed.f * d_min) + t_cur_ref;
    Eigen::Vector3f P_max = R_cur_ref * (seed.f * d_max) + t_cur_ref;

    if (P_min.z() <= 0.0f || P_max.z() <= 0.0f)
        return false;

    // Camera2Pixel 接受 Vector3d 返回 Vector2d
    Eigen::Vector2d px_min = mpCamera->Camera2Pixel(P_min.cast<double>());
    Eigen::Vector2d px_max = mpCamera->Camera2Pixel(P_max.cast<double>());

    Eigen::Vector2d epipolar_dir = px_max - px_min;
    double epipolar_length = epipolar_dir.norm();
    if (epipolar_length < 1.0)
        return false; // 视差太小，不具备测量能力

    epipolar_dir /= epipolar_length;

    // 沿极线步长搜索 NCC 最佳匹配点
    float best_ncc = -1.0f;
    Eigen::Vector2d best_pt = px_min;

    for (double l = 0.0; l <= epipolar_length; l += 0.7)
    {
        Eigen::Vector2d pt = px_min + l * epipolar_dir;
        if (pt.x() < 5 || pt.x() >= cur_img.cols - 5 || pt.y() < 5 || pt.y() >= cur_img.rows - 5)
            continue;

        float ncc = ComputeNCC(mRefImg, cur_img, seed.px, pt.cast<float>());
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_pt = pt;
        }
    }

    if (best_ncc < 0.8f) // 匹配质量不达标
        return false;

    best_curr_px = best_pt;

    // 根据几何匹配算出的三角化深度，计算对应的逆深度 tau_inv
    float match_dist = static_cast<float>((best_curr_px - px_min).norm());
    float ratio = match_dist / static_cast<float>(epipolar_length);
    float z_meas = 1.0f / ((1.0f - ratio) * (1.0f / d_min) + ratio * (1.0f / d_max));

    depth_best = 1.0f / z_meas; // 即 tau_inv

    // 根据一个像素的几何不确定性反算逆深度的测量方差 sigma2_tau
    float z_meas_px_plus = 1.0f / ((1.0f - std::min(1.0f, ratio + 1.0f / static_cast<float>(epipolar_length))) * (1.0f / d_min) +
                                   std::min(1.0f, ratio + 1.0f / static_cast<float>(epipolar_length)) * (1.0f / d_max));
    float tau_meas_px_plus = 1.0f / z_meas_px_plus;
    variance_best = (tau_meas_px_plus - depth_best) * (tau_meas_px_plus - depth_best);

    return true;
}

void DepthFilter::UpdateSeed(Seed &seed, float x, float tau_sq)
{
    // Vogiatzis 提出的 Gaussian-Uniform 混合模型贝叶斯更新规则
    float norm_pdf = (1.0f / std::sqrt(2.0f * M_PI * (seed.sigma2 + tau_sq))) *
                     std::exp(-0.5f * (x - seed.mu) * (x - seed.mu) / (seed.sigma2 + tau_sq));

    float C1 = (seed.a / (seed.a + seed.b)) * norm_pdf;
    float C2 = (seed.b / (seed.a + seed.b)) * (1.0f / seed.z_range);
    float C = C1 + C2;

    float C1_div_C = C1 / C;

    // 更新一阶矩与二阶矩
    float mu_post = C1_div_C * ((seed.sigma2 * x + tau_sq * seed.mu) / (seed.sigma2 + tau_sq)) + (1.0f - C1_div_C) * seed.mu;

    float sigma2_post = C1_div_C * ((seed.sigma2 * tau_sq) / (seed.sigma2 + tau_sq) +
                                   ((seed.sigma2 * x + tau_sq * seed.mu) / (seed.sigma2 + tau_sq)) *
                                       ((seed.sigma2 * x + tau_sq * seed.mu) / (seed.sigma2 + tau_sq))) +
                        (1.0f - C1_div_C) * (seed.sigma2 + seed.mu * seed.mu) - mu_post * mu_post;

    // 更新 Beta 分布参数
    seed.a = seed.a + C1_div_C;
    seed.b = seed.b + (1.0f - C1_div_C);

    seed.mu = mu_post;
    seed.sigma2 = std::max(0.000001f, sigma2_post);
}

float DepthFilter::ComputeNCC(const cv::Mat &ref_img, const cv::Mat &cur_img, const Eigen::Vector2f &pt_ref, const Eigen::Vector2f &pt_cur)
{
    if (ref_img.empty() || cur_img.empty())
        return 0.0f;

    int half_patch = 3;
    float mean_ref = 0.0f, mean_cur = 0.0f;

    std::vector<float> vec_ref, vec_cur;
    vec_ref.reserve(49);
    vec_cur.reserve(49);

    for (int dy = -half_patch; dy <= half_patch; ++dy)
    {
        for (int dx = -half_patch; dx <= half_patch; ++dx)
        {
            float val_r = ref_img.at<uchar>(pt_ref.y() + dy, pt_ref.x() + dx);
            float val_c = cur_img.at<uchar>(pt_cur.y() + dy, pt_cur.x() + dx);

            vec_ref.push_back(val_r);
            vec_cur.push_back(val_c);

            mean_ref += val_r;
            mean_cur += val_c;
        }
    }

    mean_ref /= 49.0f;
    mean_cur /= 49.0f;

    float numerator = 0.0f;
    float denom_r = 0.0f, denom_c = 0.0f;

    for (size_t i = 0; i < vec_ref.size(); ++i)
    {
        float r = vec_ref[i] - mean_ref;
        float c = vec_cur[i] - mean_cur;

        numerator += r * c;
        denom_r += r * r;
        denom_c += c * c;
    }

    if (denom_r * denom_c < 1e-6f)
        return 0.0f;

    return numerator / std::sqrt(denom_r * denom_c);
}

std::vector<Seed> DepthFilter::GetConvergedSeeds()
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);
    std::vector<Seed> converged_seeds;
    std::vector<Seed> active_seeds;

    for (const auto &seed : mvSeeds)
    {
        if (seed.is_converged)
        {
            converged_seeds.push_back(seed);
        }
        else
        {
            active_seeds.push_back(seed);
        }
    }

    mvSeeds = std::move(active_seeds);
    return converged_seeds;
}