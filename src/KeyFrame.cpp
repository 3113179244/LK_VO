#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Camera.h"

KeyFrame::KeyFrame(std::shared_ptr<Frame> pFrame)
    : mId(pFrame->mFrameId), 
      mFrameId(pFrame->mFrameId), 
      mTimeStamp(pFrame->mTimeStamp), 
      N(pFrame->iFeaturePointnums), 
      mvKeysLeft(pFrame->mvleftpixel), 
      mvpMapPoints(pFrame->mvpMapPoints)
{
    SetPose(pFrame->GetPose());
}

void KeyFrame::SetPose(const Eigen::Matrix4f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw;
    Eigen::Matrix3f Rcw = mTcw.block<3, 3>(0, 0);
    Eigen::Vector3f tcw = mTcw.block<3, 1>(0, 3);
    mOw = -Rcw.transpose() * tcw;
}

Eigen::Matrix4f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw;
}

Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOw;
}

std::vector<std::shared_ptr<MapPoint>> KeyFrame::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

std::shared_ptr<MapPoint> KeyFrame::GetMapPoint(const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

void KeyFrame::EraseMapPointMatch(std::shared_ptr<MapPoint> pMP)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    for (size_t i = 0; i < mvpMapPoints.size(); i++)
    {
        if (mvpMapPoints[i] == pMP)
        {
            mvpMapPoints[i] = nullptr;
            break;
        }
    }
}

void KeyFrame::AddConnection(std::shared_ptr<KeyFrame> pKF, const int weight)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
        mConnectedKeyFrameWeights[pKF] = weight;
    else
        mConnectedKeyFrameWeights[pKF] += weight;
}

void KeyFrame::EraseConnection(std::shared_ptr<KeyFrame> pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        mConnectedKeyFrameWeights.erase(pKF);
}

std::map<std::shared_ptr<KeyFrame>, int> KeyFrame::GetConnectedKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mConnectedKeyFrameWeights;
}