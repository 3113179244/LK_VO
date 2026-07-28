#include "DepthFilter.h"
#include "Camera.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>

DepthFilter::DepthFilter(std::shared_ptr<Camera> pCamera, float baseline, float max_disparity)
    : mpCamera(pCamera), mBaseline(baseline), mMaxDisparity(max_disparity), mNextSeedId(0)
{
    // 假设相机内参 fx ≈ fy
    mFocalLength = static_cast<float>(pCamera->fx);
}

void DepthFilter::AddSeed(const Eigen::Vector2f &pt_left, float init_depth, float min_depth)
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);
    Eigen::Vector3f ray = mpCamera->Pixel2Camera(pt_left.cast<double>()).cast<float>();
    mvSeeds.emplace_back(mNextSeedId++, pt_left, ray, init_depth, min_depth);
}

void DepthFilter::Update(const cv::Mat &left_img, const cv::Mat &right_img)
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);

    for (auto &seed : mvSeeds)
    {
        if (seed.is_converged)
            continue;

        float tau_obs = 0.0f, sigma2_obs = 0.0f;
        if (ComputeObservation(seed, left_img, right_img, tau_obs, sigma2_obs))
        {
            UpdateSeed(seed, tau_obs, sigma2_obs);

            // 收敛判定（同单目）
            if (std::sqrt(seed.sigma2) < 0.005f * seed.mu || seed.a / (seed.a + seed.b) > 0.85f)
            {
                seed.is_converged = true;
            }
        }
    }
}

bool DepthFilter::ComputeObservation(const Seed &seed,
                                                 const cv::Mat &left_img,
                                                 const cv::Mat &right_img,
                                                 float &tau_obs,
                                                 float &sigma2_obs)
{
    // 左目特征点
    cv::Point2f ptL(seed.px.x(), seed.px.y());

    // 粗略估计当前深度对应的视差：d = f * baseline / depth
    float depth_est = 1.0f / seed.mu;
    float disp_est = mFocalLength * mBaseline / depth_est;

    // 搜索范围：以 disp_est 为中心，±3σ 范围（σ 来自逆深度的方差转化）
    float sigma_disp = mFocalLength * mBaseline * std::sqrt(seed.sigma2); // 近似
    float disp_min = std::max(0.0f, disp_est - 3.0f * sigma_disp);
    float disp_max = std::min(mMaxDisparity, disp_est + 3.0f * sigma_disp);

    if (disp_max - disp_min < 1.0f)
        return false; // 搜索范围太小

    // 准备左目参考块
    int half = mPatchSize / 2;
    cv::Rect roiL(ptL.x - half, ptL.y - half, mPatchSize, mPatchSize);
    if (roiL.x < 0 || roiL.y < 0 || roiL.x + mPatchSize > left_img.cols || roiL.y + mPatchSize > left_img.rows)
        return false;

    cv::Mat patchL = left_img(roiL).clone();

    float best_ncc = -1.0f;
    int best_disp = -1;

    // 沿水平极线搜索（只在右目）
    for (int d = static_cast<int>(disp_min); d <= static_cast<int>(disp_max); ++d)
    {
        cv::Point2f ptR(ptL.x - d, ptL.y); // 注意：双目校正后，右目对应点 x = x_left - d
        cv::Rect roiR(ptR.x - half, ptR.y - half, mPatchSize, mPatchSize);
        if (roiR.x < 0 || roiR.y < 0 || roiR.x + mPatchSize > right_img.cols || roiR.y + mPatchSize > right_img.rows)
            continue;

        cv::Mat patchR = right_img(roiR).clone();
        float ncc = ComputeNCC(patchL, patchR);
        if (ncc > best_ncc)
        {
            best_ncc = ncc;
            best_disp = d;
        }
    }

    if (best_ncc < 0.6f || best_disp < 0)
        return false; // 匹配质量不佳

    // 由视差计算深度
    float depth_obs = mFocalLength * mBaseline / static_cast<float>(best_disp);
    tau_obs = 1.0f / depth_obs;

    // 观测方差：假设视差匹配误差为 ±0.5 像素（可调）
    float disp_err = 0.5f;
    float depth_err = mFocalLength * mBaseline * disp_err / (best_disp * best_disp); // 对 d 求导
    sigma2_obs = (1.0f / (depth_obs - depth_err) - tau_obs) * (1.0f / (depth_obs - depth_err) - tau_obs);

    return true;
}

void DepthFilter::UpdateSeed(Seed &seed, float tau_obs, float sigma2_obs)
{
    // 与单目 DepthFilter::UpdateSeed 完全一致
    float norm_pdf = (1.0f / std::sqrt(2.0f * M_PI * (seed.sigma2 + sigma2_obs))) *
                     std::exp(-0.5f * (tau_obs - seed.mu) * (tau_obs - seed.mu) / (seed.sigma2 + sigma2_obs));

    float C1 = (seed.a / (seed.a + seed.b)) * norm_pdf;
    float C2 = (seed.b / (seed.a + seed.b)) * (1.0f / seed.z_range);
    float C = C1 + C2;
    float C1_div_C = C1 / C;

    float mu_post = C1_div_C * ((seed.sigma2 * tau_obs + sigma2_obs * seed.mu) / (seed.sigma2 + sigma2_obs)) +
                    (1.0f - C1_div_C) * seed.mu;

    float sigma2_post = C1_div_C * ((seed.sigma2 * sigma2_obs) / (seed.sigma2 + sigma2_obs) +
                                    ((seed.sigma2 * tau_obs + sigma2_obs * seed.mu) / (seed.sigma2 + sigma2_obs)) *
                                        ((seed.sigma2 * tau_obs + sigma2_obs * seed.mu) / (seed.sigma2 + sigma2_obs))) +
                        (1.0f - C1_div_C) * (seed.sigma2 + seed.mu * seed.mu) - mu_post * mu_post;

    seed.a = seed.a + C1_div_C;
    seed.b = seed.b + (1.0f - C1_div_C);
    seed.mu = mu_post;
    seed.sigma2 = std::max(0.000001f, sigma2_post);
}

float DepthFilter::ComputeNCC(const cv::Mat &patch_ref, const cv::Mat &patch_cur)
{
    cv::Mat ref_f, cur_f;
    patch_ref.convertTo(ref_f, CV_32F);
    patch_cur.convertTo(cur_f, CV_32F);

    cv::Scalar mean_ref = cv::mean(ref_f);
    cv::Scalar mean_cur = cv::mean(cur_f);
    ref_f -= mean_ref[0];
    cur_f -= mean_cur[0];

    float numerator = static_cast<float>(ref_f.dot(cur_f));
    float denom = static_cast<float>(std::sqrt(ref_f.dot(ref_f) * cur_f.dot(cur_f)));

    if (denom < 1e-6f)
        return 0.0f;
    return numerator / denom;
}

std::vector<Seed> DepthFilter::GetConvergedSeeds()
{
    std::unique_lock<std::mutex> lock(mMutexSeeds);
    std::vector<Seed> converged;
    std::vector<Seed> active;

    for (auto &s : mvSeeds)
    {
        if (s.is_converged)
            converged.push_back(s);
        else
            active.push_back(s);
    }
    mvSeeds = std::move(active);
    return converged;
}