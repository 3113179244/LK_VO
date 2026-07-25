#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Camera.h"

KeyFrame::KeyFrame(std::shared_ptr<Frame> pFrame)
    : mId(pFrame->mId), mFrameId(pFrame->mId), mTimeStamp(pFrame->mTimeStamp),
      mpCamera(pFrame->mpCamera), mbf(pFrame->mbf), N(pFrame->N),
      mvKeysLeft(pFrame->mvKeys), mvpMapPoints(pFrame->mvpMapPoints)
{
    SetPose(pFrame->GetPose());
}

void KeyFrame::SetPose(const cv::Mat &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw.clone();
    cv::Mat Rcw = mTcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = mTcw.rowRange(0, 3).col(3);
    mOw = -Rcw.t() * tcw;
}

cv::Mat KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOw.clone();
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