#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include <algorithm>
#include "ORBextractor.h"
long unsigned int KeyFrame::nNextId = 0;

// 从 Frame 拷贝并生成 KeyFrame
KeyFrame::KeyFrame(Frame &F, Map *pMap)
    : mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp),
      fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
      mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), mK(F.mK.clone()),
      N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn), mvuRight(F.mvuRight), mvDepth(F.mvDepth),
      mDescriptors(F.mDescriptors.clone()),
      mnScaleLevels(F.mpORBextractorLeft->GetLevels()), mfScaleFactor(F.mpORBextractorLeft->GetScaleFactor()),
      mvScaleFactors(F.mpORBextractorLeft->GetScaleFactors()),
      mvLevelSigma2(F.mpORBextractorLeft->GetScaleSigmaSquares()),
      mvInvLevelSigma2(F.mpORBextractorLeft->GetInverseScaleSigmaSquares()),
      mbBad(false), mpMap(pMap), mpORBvocabulary(F.mpORBvocabulary)
{
    mnId = nNextId++;
    mvpMapPoints = F.mvpMapPoints;
    SetPose(F.mTcw); // 此处的 F.mTcw 也对应改成了 Eigen::Matrix4f
}

// 线程安全地设置位姿
void KeyFrame::SetPose(const Eigen::Matrix4f &Tcw_)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Tcw = Tcw_;
    Rcw = Tcw.block<3, 3>(0, 0);
    tcw = Tcw.block<3, 1>(0, 3);
    Rwc = Rcw.transpose();
    Ow = -Rwc * tcw;
}

Eigen::Matrix4f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw;
}

Eigen::Matrix4f KeyFrame::GetPoseInverse()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
    Twc.block<3, 3>(0, 0) = Rwc;
    Twc.block<3, 1>(0, 3) = Ow;
    return Twc;
}

Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Ow;
}

Eigen::Matrix3f KeyFrame::GetRotation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Rcw;
}

Eigen::Vector3f KeyFrame::GetTranslation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return tcw;
}

void KeyFrame::UpdateConnections()
{
    std::map<KeyFrame *, int> KFcounter;
    std::vector<MapPoint *> vpMP;

    {
        std::unique_lock<std::mutex> lockMatches(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (!pMP || pMP->isBad())
            continue;
    }

    if (KFcounter.empty())
        return;

    int nmax = 0;
    KeyFrame *pKFmax = nullptr;
    int th = 15;

    std::vector<std::pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(KFcounter.size());

    for (auto mit = KFcounter.begin(); mit != KFcounter.end(); mit++)
    {
        if (mit->second > nmax)
        {
            nmax = mit->second;
            pKFmax = mit->first;
        }
        if (mit->second >= th)
        {
            vPairs.push_back(std::make_pair(mit->second, mit->first));
        }
    }

    if (vPairs.empty())
    {
        vPairs.push_back(std::make_pair(nmax, pKFmax));
    }

    std::sort(vPairs.begin(), vPairs.end());
    std::vector<KeyFrame *> vNeighbors;
    vNeighbors.reserve(vPairs.size());
    std::vector<int> vWeights;
    vWeights.reserve(vPairs.size());

    for (size_t i = 0; i < vPairs.size(); i++)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vNeighbors;
        mvOrderedWeights = vWeights;
    }
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
        mConnectedKeyFrameWeights[pKF] = weight;
    else if (mConnectedKeyFrameWeights[pKF] != weight)
        mConnectedKeyFrameWeights[pKF] = weight;
    else
        return;

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    std::vector<std::pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());

    for (auto mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
        vPairs.push_back(std::make_pair(mit->second, mit->first));

    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame *> vNeighbors;
    vNeighbors.reserve(vPairs.size());
    std::vector<int> vWeights;
    vWeights.reserve(vPairs.size());

    for (int i = vPairs.size() - 1; i >= 0; i--)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    mvpOrderedConnectedKeyFrames = vNeighbors;
    mvOrderedWeights = vWeights;
}

std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx >= 0)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }
}

void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}