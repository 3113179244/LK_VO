#include "Frame.h"
#include "MapPoint.h"
#include "Camera.h"
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double timestamp,
             std::shared_ptr<Camera> camera, const int id)
    : mId(id), mTimeStamp(timestamp), mpCamera(camera),
      mTcw(cv::Mat::eye(4, 4, CV_32F))
{

    mbf = camera->mBaseline * camera->fx;
    mThDepth = camera->mBaseline * 35.0f;
}

void Frame::ExtractFeatures()
{
}

void Frame::ComputeStereoMatches()
{
    // 此时假设 mvuRight 已经由光流前端赋值
    // mvKeys 也已经赋值
    mvDepth.resize(N, -1.0f);

    for (int i = 0; i < N; i++)
    {
        float u_left = mvKeys[i].pt.x;
        float u_right = mvuRight[i];

        // 如果右目成功匹配到
        if (u_right > 0)
        {
            float disparity = u_left - u_right;

            // 视差必须大于0才有效
            if (disparity > 0.0f)
            {
                float depth = mbf / disparity;

                // ==========================================
                // 远近点判断核心逻辑
                // ==========================================
                if (depth > mThDepth)
                {
                    // 1. 远点处理
                    // 深度超过阈值，视差太小导致深度极不可靠。
                    // 强行丢弃深度和右目匹配，交由后端按单目模式（仅提供方向向量）优化。
                    mvDepth[i] = -1.0f;
                    mvuRight[i] = -1.0f;
                }
                else
                {
                    // 2. 近点处理
                    // 深度在可靠范围内，保留具体的深度值，提供 3D 约束
                    mvDepth[i] = depth;
                }
            }
            else
            {
                // 视差异常 (如极线未完全对齐导致右侧点跑到左侧)
                mvDepth[i] = -1.0f;
                mvuRight[i] = -1.0f;
            }
        }
    }
}

void Frame::SetPose(const cv::Mat &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw.clone();
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mtcw = mTcw.rowRange(0, 3).col(3);
    mRwc = mRcw.t();
    mOw = -mRwc * mtcw;
}

cv::Mat Frame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw.clone();
}

cv::Mat Frame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOw.clone();
}

bool Frame::isInFrustum(const MapPoint *pMP, float viewingCosLimit)
{
    // 投影到图像平面并检查边界和视角的具体逻辑
    return true;
}