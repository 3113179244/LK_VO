#include "LocalMapping.h"
#include "System.h"
#include "Map.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Tracker.h"

#include <unistd.h> // usleep
#include <algorithm>

LocalMapping::LocalMapping(System *pSys, std::shared_ptr<Map> pMap)
    : mpSystem(pSys), mpMap(pMap), mpTracker(nullptr),
      mpCurrentKeyFrame(nullptr), mbStopRequested(false),
      mbStopped(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
    // 启动 LocalMapping 独立线程
    mpThread = new std::thread(&LocalMapping::Run, this);
}

LocalMapping::~LocalMapping()
{
    if (mpThread)
    {
        mpThread->join();
        delete mpThread;
    }
}

void LocalMapping::Run()
{
    mbStopped = false;

    while (1)
    {
        // 告知 Tracking 线程，LocalMapping 处于忙碌状态，暂时不要频繁插入关键帧
        SetNotStop();

        // 检查是否有新的关键帧等待处理
        if (CheckNewKeyFrames())
        {
            // 1. 处理队列中的首个关键帧（关联 MapPoint，更新共视图 Connections）
            ProcessNewKeyFrame();

            // 2. 剔除近期新增的质量不佳的 MapPoints（观测数不足或可见率低）
            MapPointCulling();

            // 3. 通过与相邻关键帧三角化，创建新的 MapPoints
            CreateNewMapPoints();

            // 4. 融合邻近关键帧中重复的 MapPoints
            SearchInNeighbors();

            // 5. 执行 Local BA 优化 (这里可调用 Optimizer::LocalBundleAdjustment)
            // if (!CheckNewKeyFrames()) { Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, mpMap); }

            // 6. 剔除冗余的关键帧（如果某关键帧 90% 以上的地图点能被其他至少3个关键帧看到）
            KeyFrameCulling();
        }

        // 处理完毕，解除 NotStop 标记
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            mbNotStop = false;
        }

        // 检查外部是否有停止请求（如回环检测 LoopClosing 触发 pause）
        if (GetStopRequired())
        {
            std::unique_lock<std::mutex> lock(mMutexStop);
            mbStopped = true;
            while (isStopped())
            {
                usleep(3000); // 挂起线程
            }
        }

        usleep(3000); // 适当休眠，避免 CPU 空转
    }
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    mlNewKeyFrames.push_back(pKF);
}

bool LocalMapping::CheckNewKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
    return !mlNewKeyFrames.empty();
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        std::unique_lock<std::mutex> lock(mMutexNewKeyBase);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // 1. 关联普通帧匹配生成的地图点到关键帧，并加入局部待检验列表
    std::vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
    {
        MapPoint *pMP = vpMapPointMatches[i];
        if (pMP && !pMP->isBad())
        {
            if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
            {
                pMP->AddObservation(mpCurrentKeyFrame, i);
                pMP->UpdateNormalAndDepth();
                pMP->ComputeDistinctiveDescriptor();
            }
            else
            {
                // 将近期创建的地图点放入考核队列
                mlpRecentAddedMapPoints.push_back(pMP);
            }
        }
    }

    // 2. 更新关键帧在共视图（Covisibility Graph）中的连接关系
    mpCurrentKeyFrame->UpdateConnections();

    // 3. 将关键帧插入全局地图
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // 对近期创建的地图点执行严格的质量考核：
    // 条件1: 被标记为 Bad 的直接从列表中剔除
    // 条件2: 被观测到的实际比例 (Found / Visible) < 25% 的判定为坏点并剔除
    // 条件3: 从创建起经过了连续 2 个关键帧后，观测到它的关键帧数量 < 2 (单目) 或 < 3 (双目) 则剔除
    auto lit = mlpRecentAddedMapPoints.begin();
    while (lit != mlpRecentAddedMapPoints.end())
    {
        MapPoint *pMP = *lit;
        if (pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (pMP->GetFoundRatio() < 0.25f)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)mpCurrentKeyFrame->mnId - (int)pMP->mnId) >= 2 && pMP->GetObservations().size() <= 2)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (((int)mpCurrentKeyFrame->mnId - (int)pMP->mnId) >= 3)
        {
            // 通过考核，不再是“近期新增点”，从待考列表中移除
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else
        {
            lit++;
        }
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // 搜集当前关键帧共视程度最高的前 10/20 个邻居关键帧，通过特征匹配与对极几何/三角化生成新的 MapPoints
    std::vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(10);
    // ... 执行三角化匹配逻辑并 mpMap->AddMapPoint(...)
}

void LocalMapping::SearchInNeighbors()
{
    // 融合当前关键帧及其邻近关键帧中重复的地图点（通过投影匹配、距离阈值融合）
    // ... 调用 ORBmatcher 进行二次融合与 MapPoint::Replace 替换
}

// LocalMapping.cpp
void LocalMapping::KeyFrameCulling()
{
    // 替换为已有接口名：GetConnectedKeyFrames()
    std::vector<KeyFrame *> vpConnectedKeyFrames = mpCurrentKeyFrame->GetConnectedKeyFrames();

    for (auto pKF : vpConnectedKeyFrames)
    {
        if (pKF->mnId == 0)
            continue; // 保留初始化帧

        std::vector<MapPoint *> vpMapPoints = pKF->GetMapPointMatches();
        int nRedundantObservations = 0;
        int nTotalObservations = 0;

        for (size_t i = 0; i < vpMapPoints.size(); i++)
        {
            MapPoint *pMP = vpMapPoints[i];
            if (pMP && !pMP->isBad())
            {
                nTotalObservations++;
                if (pMP->GetObservations().size() > 3)
                {
                    nRedundantObservations++;
                }
            }
        }

        if (nTotalObservations > 0 && (float)nRedundantObservations / nTotalObservations > 0.90f)
        {
            pKF->SetBadFlag(); // 现在已可正常调用
        }
    }
}

// 线程控制相关函数
void LocalMapping::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool LocalMapping::GetStopRequired()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool LocalMapping::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::SetNotStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (mbStopped)
        return false;
    mbNotStop = true;
    return true;
}