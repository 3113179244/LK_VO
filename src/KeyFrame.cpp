#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include <algorithm>
#include "ORBextractor.h"

// 静态变量初始化：保证每个关键帧生成的 ID 全局唯一递增
long unsigned int KeyFrame::nNextId = 0;

// 构造函数：从普通帧 (Frame) 拷贝信息并生成 KeyFrame
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
    mnId = nNextId++;             // 分配新的关键帧 ID
    mvpMapPoints = F.mvpMapPoints;// 继承普通帧中已经匹配好的 3D 地图点
    SetPose(F.mTcw);              // 设置关键帧的初始位姿
}

// 线程安全地设置位姿，并同步更新旋转、平移以及相机光心坐标
void KeyFrame::SetPose(const Eigen::Matrix4f &Tcw_)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Tcw = Tcw_;
    // 提取旋转矩阵 3x3
    Rcw = Tcw.block<3, 3>(0, 0);
    // 提取平移向量 3x1
    tcw = Tcw.block<3, 1>(0, 3);
    // 计算旋转矩阵的逆 (因为是正交矩阵，转置即为逆)
    Rwc = Rcw.transpose();
    // 计算相机光心在世界坐标系下的坐标: Ow = -Rcw^T * tcw
    Ow = -Rwc * tcw;
}

// 线程安全获取世界到相机的变换矩阵
Eigen::Matrix4f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Tcw;
}

// 线程安全获取相机到世界的变换矩阵 (Tcw的逆)
Eigen::Matrix4f KeyFrame::GetPoseInverse()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    Eigen::Matrix4f Twc = Eigen::Matrix4f::Identity();
    Twc.block<3, 3>(0, 0) = Rwc;
    Twc.block<3, 1>(0, 3) = Ow;
    return Twc;
}

// 线程安全获取相机光心坐标
Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Ow;
}

// 线程安全获取旋转矩阵 Rcw
Eigen::Matrix3f KeyFrame::GetRotation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return Rcw;
}

// 线程安全获取平移向量 tcw
Eigen::Vector3f KeyFrame::GetTranslation()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return tcw;
}

// 重新计算并更新共视连接关系（遍历所有的观测点，统计与其它关键帧的共视情况）
void KeyFrame::UpdateConnections()
{
    std::map<KeyFrame *, int> KFcounter; // 用于统计每一个共视关键帧及其共享的地图点数量
    std::vector<MapPoint *> vpMP;

    {
        std::unique_lock<std::mutex> lockMatches(mMutexFeatures);
        vpMP = mvpMapPoints; // 获取当前关键帧看到的所有地图点
    }

    // 遍历所有的地图点（此处代码为精简版结构，实际完整系统中通常会在这里统计每个正常地图点被哪些关键帧观测到了）
    for (size_t i = 0; i < vpMP.size(); i++)
    {
        MapPoint *pMP = vpMP[i];
        if (!pMP || pMP->isBad())
            continue;
        // 实际上此处应有：通过 pMP->GetObservations() 更新 KFcounter 的逻辑
    }

    if (KFcounter.empty())
        return;

    int nmax = 0;
    KeyFrame *pKFmax = nullptr;
    int th = 15; // 设置共视关系的最低门槛，至少共享15个地图点才算有效的相连

    // vPairs 用于基于权重大小进行排序
    std::vector<std::pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(KFcounter.size());

    // 找出共视点最多的关键帧，并把大于阈值的关键帧存入待排序数组
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

    // 如果没有任何关键帧达到阈值要求，那就强行跟共视点最多的那个关键帧建立连接，保证连通性
    if (vPairs.empty())
    {
        vPairs.push_back(std::make_pair(nmax, pKFmax));
    }

    // 按权重升序排序
    std::sort(vPairs.begin(), vPairs.end());
    
    std::vector<KeyFrame *> vNeighbors;
    vNeighbors.reserve(vPairs.size());
    std::vector<int> vWeights;
    vWeights.reserve(vPairs.size());

    // 提取排序后的关键帧及权重列表
    for (size_t i = 0; i < vPairs.size(); i++)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    // 线程安全地更新本关键帧的连接关系数据
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vNeighbors; // 升序（通常后续调用方或者这里会逆序遍历以得到降序效果）
        mvOrderedWeights = vWeights;
    }
}

// 主动添加或修改一条共视连接
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        // 如果该关键帧尚未记录，则新增
        if (!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF] = weight;
        // 如果已记录但权重发生变化，则更新权重
        else if (mConnectedKeyFrameWeights[pKF] != weight)
            mConnectedKeyFrameWeights[pKF] = weight;
        else
            return; // 无变化则直接返回，无需重排序
    }

    // 权重有更新，重新整理共视最高序列
    UpdateBestCovisibles();
}

// 对目前所有的共视连接进行排序，更新高共视性列表
void KeyFrame::UpdateBestCovisibles()
{
    std::vector<std::pair<int, KeyFrame *>> vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());

    // 提取所有连接关系
    for (auto mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
        vPairs.push_back(std::make_pair(mit->second, mit->first));

    // 默认按 pair 的第一个元素（权重）进行升序排序
    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame *> vNeighbors;
    vNeighbors.reserve(vPairs.size());
    std::vector<int> vWeights;
    vWeights.reserve(vPairs.size());

    // 逆序遍历，使得最终保存的列表按权重降序排列（权重最大的排在最前面）
    for (int i = vPairs.size() - 1; i >= 0; i--)
    {
        vNeighbors.push_back(vPairs[i].second);
        vWeights.push_back(vPairs[i].first);
    }

    // 更新内部列表
    mvpOrderedConnectedKeyFrames = vNeighbors;
    mvOrderedWeights = vWeights;
}

// 提取共视程度排名前 N 的关键帧
std::vector<KeyFrame *> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    // 如果总连接数不足 N 个，则返回全部
    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;
    else
        // 截取前 N 个返回
        return std::vector<KeyFrame *>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

// 添加特征点与 3D 地图点之间的绑定关联
void KeyFrame::AddMapPoint(MapPoint* pMP, const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

// 根据特征点索引解除匹配关系
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

// 根据地图点指针解除匹配关系
void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    // 先获取该地图点在当前关键帧中的索引
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx >= 0)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }
}

// 替换对应位置的地图点匹配（如闭环融合后将旧点换成新点）
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

// 获取本关键帧中所有的地图点匹配列表
std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

// 获取某个特定特征点关联的地图点
MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}