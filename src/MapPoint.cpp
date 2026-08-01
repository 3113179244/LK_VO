#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"

long unsigned int MapPoint::nNextId = 0;
std::mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const Eigen::Vector3f &Pos, KeyFrame* pRefKF, Map* pMap)
    : mWorldPos(Pos), mpRefKF(pRefKF), mpMap(pMap), mnVisible(1), mnFound(1), mbBad(false), mpReplaced(nullptr)
{
    mnId = nNextId++;
    mNormalVector.setZero();
}

void MapPoint::SetWorldPos(const Eigen::Vector3f &Pos)
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    mWorldPos = Pos;
}

Eigen::Vector3f MapPoint::GetWorldPos()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mWorldPos;
}

Eigen::Vector3f MapPoint::GetNormal()
{
    std::unique_lock<std::mutex> lock(mMutexPos);
    return mNormalVector;
}

void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return;
    mObservations[pKF] = idx;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            mObservations.erase(pKF);
            if(mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // 当观测到该点的关键帧少于 2 个时，标记为坏点
            if(mObservations.size() <= 1)
                bBad = true;
        }
    }

    if(bBad)
        SetBadFlag();
}

std::map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations;
}

// 计算代表性描述子：找到与其他所有观测描述子汉明距离中位数最小的那个描述子
void MapPoint::ComputeDistinctiveDescriptor()
{
    std::vector<cv::Mat> vDescriptors;
    std::map<KeyFrame*, size_t> observations;

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        if(mbBad) return;
        observations = mObservations;
    }

    if(observations.empty()) return;

    vDescriptors.reserve(observations.size());

    for(auto mit = observations.begin(); mit != observations.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        if(!pKF->mbBad)
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty()) return;

    // 算两两之间的距离矩阵
    const size_t N = vDescriptors.size();
    float Distances[N][N];
    for(size_t i = 0; i < N; i++)
    {
        Distances[i][i] = 0;
        for(size_t j = i + 1; j < N; j++)
        {
            int dist = cv::norm(vDescriptors[i], vDescriptors[j], cv::NORM_HAMMING);
            Distances[i][j] = dist;
            Distances[j][i] = dist;
        }
    }

    int BestMedian = INT_MAX;
    int BestIdx = 0;
    for(size_t i = 0; i < N; i++)
    {
        std::vector<int> vDists(Distances[i], Distances[i] + N);
        std::sort(vDists.begin(), vDists.end());
        int median = vDists[0.5 * (N - 1)];

        if(median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

// 更新平均观测方向向量与尺度（距离）下限/上限
void MapPoint::UpdateNormalAndDepth()
{
    std::map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    Eigen::Vector3f Pos;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        if(mbBad) return;
        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos;
    }

    if(observations.empty()) return;

    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    int n=0;

    for(auto mit = observations.begin(); mit != observations.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        Eigen::Vector3f Owc = pKF->GetCameraCenter();
        Eigen::Vector3f normali = Pos - Owc;
        normal += normali.normalized();
        n++;
    }

    Eigen::Vector3f Owc = pRefKF->GetCameraCenter();
    Eigen::Vector3f dist = Pos - Owc;
    const float distRef = dist.norm();

    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor = pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        mfMinDistance = distRef / levelScaleFactor;
        mfMaxDistance = mfMinDistance * pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal.normalized();
    }
}

void MapPoint::SetBadFlag()
{
    std::map<KeyFrame*, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        mbBad = true;
        obs = mObservations;
        mObservations.clear();
    }
    for(auto mit = obs.begin(); mit != obs.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    std::unique_lock<std::mutex> lock1(mMutexFeatures);
    std::unique_lock<std::mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId == this->mnId) return;

    std::map<KeyFrame*, size_t> obs;
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        obs = mObservations;
        mObservations.clear();
        mbBad = true;
        mpReplaced = pMP;
    }

    for(auto mit = obs.begin(); mit != obs.end(); mit++)
    {
        KeyFrame* pKF = mit->first;
        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF, mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }

    pMP->ComputeDistinctiveDescriptor();
    pMP->UpdateNormalAndDepth();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::IsInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mObservations.count(pKF);
}

int MapPoint::GetIndexInKeyFrame(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    if (mObservations.count(pKF)) {
        return mObservations[pKF];
    }
    return -1;
}