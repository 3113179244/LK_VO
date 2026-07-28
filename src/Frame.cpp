#include "Frame.h"
#include "Frame.h"
#include "Camera.h"
#include "MapPoint.h"
Frame::Frame(const double dtimestamp, const cv::Mat &image0, const cv::Mat &image1, const int FrameId)
    : mFrameId(FrameId), mTimeStamp(dtimestamp), mTcw(Eigen::Matrix4f::Identity()) 
{

}

void Frame::SetPose(const Eigen::Matrix4f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw;

    mRcw = mTcw.block<3, 3>(0, 0);
    mtcw = mTcw.block<3, 1>(0, 3);

    mRwc = mRcw.transpose(); 
    mOw = -mRwc * mtcw;
}

Eigen::Matrix4f Frame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw; 
}

Eigen::Vector3f Frame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOw;
}

bool Frame::isInFrustum(const MapPoint *pMapPoint, float viewingCosLimit)
{
    return true;
}