#include "Tracker.h"
#include "Config.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FrameDrawer.h"
#include <algorithm>
#include <iostream>
#include "ORBmatcher.h"
#include "MotionOnlyBA.h"
#include "LocalMapping.h"
Tracker::Tracker(System *pSys, std::shared_ptr<Map> pMap, int sensor)
    : mpSystem(pSys), mpMap(pMap), mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()), mpReferenceKF(nullptr), mpLocalMapper(nullptr)
{
    // 从 Config 类中加载 ORB 提取器参数
    int nFeatures = Config::g_nORBnFeatures;
    float fScaleFactor = Config::g_dORBscaleFactor;
    int nLevels = Config::g_nORBnLevels;
    int finiThFAST = Config::g_nORBiniThFAST;
    int fminThFAST = Config::g_nORBminThFAST;

    // 初始化左右图 ORB 提取器
    mpORBextractorLeft = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
    mpORBextractorRight = std::make_unique<ORBextractor>(nFeatures, fScaleFactor, nLevels, finiThFAST, fminThFAST);
}

Tracker::~Tracker() {}

Eigen::Matrix4f Tracker::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    // 构建内参矩阵与畸变矩阵
    cv::Mat K = (cv::Mat_<float>(3, 3) << Config::g_dFx, 0, Config::g_dCx,
                 0, Config::g_dFy, Config::g_dCy,
                 0, 0, 1);
    cv::Mat DistCoef = (cv::Mat_<float>(4, 1) << Config::g_dK1, Config::g_dK2, Config::g_dP1, Config::g_dP2);

    // 实例化当前帧 (内部自动触发多线程特征提取 + 双目匹配计算深度)
    mCurrentFrame = Frame(imRectLeft, imRectRight, timestamp,
                          mpORBextractorLeft.get(), mpORBextractorRight.get(),
                          nullptr, K, DistCoef, Config::g_dBf, Config::g_dThDepth);
    // 打印左右目图像的特征点，双目匹配成功点数
    // int nLeft = mCurrentFrame.mvKeys.size();
    // int nRight = mCurrentFrame.mvKeysRight.size();
    // int nMatches = std::count_if(mCurrentFrame.mvuRight.begin(), mCurrentFrame.mvuRight.end(),
    //                              [](float d)
    //                              { return d >= 0; }); // 或 mvDepth[i] > 0

    // std::cout << "Frame " << mCurrentFrame.mnId
    //           << " | Left features: " << nLeft
    //           << " | Right features: " << nRight
    //           << " | Stereo matches: " << nMatches << std::endl;
    // 执行跟踪状态机主逻辑
    Track();

    // 返回当前帧姿态
    return mCurrentFrame.mTcw;
}

void Tracker::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // 阶段 A: 未初始化状态 -> 执行双目初始化
    if (mState == NOT_INITIALIZED)
    {
        if (StereoInitialization())
        {
            mState = OK;
        }
        return;
    }

    // 阶段 B: 正常跟踪状态 -> 估计姿态
    bool bOK = false;

    // 1. 优先尝试恒速模型跟踪 (Velocity Model)
    if (!mVelocity.isIdentity() && mLastFrame.mnId == mCurrentFrame.mnId - 1)
    {
        bOK = TrackWithMotionModel();
    }

    // 2. 若运动模型失效，回退到参考关键帧跟踪
    if (!bOK)
    {
        bOK = TrackReferenceKeyFrame();
    }

    // 3. 跟踪局部地图进行位姿精确优化
    if (bOK)
    {
        bOK = TrackLocalMap();
    }

    if (bOK)
    {
        mState = OK;
        mVelocity = mCurrentFrame.mTcw * mLastFrame.mTcw.inverse();
        if (NeedNewKeyFrame())
        {
            CreateNewKeyFrame();
        }
    }
    else
    {
        mState = LOST;
    }
    mLastFrame = Frame(mCurrentFrame);
    if (mpFrameDrawer)
    {
        mpFrameDrawer->Update(this);
    }
}

bool Tracker::StereoInitialization()
{
    if (mCurrentFrame.N < 500)
        return false;

    mCurrentFrame.SetPose(Eigen::Matrix4f::Identity());

    // 创建第一帧对应的 KeyFrame 并加入 Map
    KeyFrame *pKFinit = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKFinit);

    // 为当前帧所有有效的双目特征点反投影生成 MapPoint
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if (z > 0)
        {
            Eigen::Vector3f p3D = mCurrentFrame.UnprojectStereo(i);
            MapPoint *pMP = new MapPoint(p3D, pKFinit, mpMap.get());

            pMP->AddObservation(pKFinit, i);
            pKFinit->AddMapPoint(pMP, i);
            pMP->ComputeDistinctiveDescriptor();
            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mCurrentFrame.mvpMapPoints[i] = pMP;
        }
    }

    mpReferenceKF = pKFinit;
    mnLastKeyFrameId = mCurrentFrame.mnId;

    return true;
}

bool Tracker::TrackWithMotionModel()
{
    // 1. 根据恒速模型粗略预测当前位姿: T_cw = V * T_lw
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // 2. 清空当前帧的地图点指针数组
    mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));

    // 3. 利用投影建立上一帧地图点与当前帧特征点的匹配
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 15); // 搜索半径阈值设为15

    if (nmatches < 20)
    {
        // 匹配点太少，放大搜索半径重试一次
        mCurrentFrame.mvpMapPoints = std::vector<MapPoint *>(mCurrentFrame.N, static_cast<MapPoint *>(nullptr));
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * 15);
    }

    if (nmatches < 20)
        return false;

    // 4. 只优化当前帧位姿 (Pose Optimization / Motion-only BA)
    int nInliers = MotionOnlyBA::Optimize(&mCurrentFrame);

    // 5. 剔除优化时被判定为外点的匹配
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(nullptr);
                mCurrentFrame.mvbOutlier[i] = false;
                nmatches--;
            }
        }
    }

    mnMatchesInliers = nInliers;

    // 6. 内点数满足要求则跟踪成功
    return (nInliers >= 10);
}

bool Tracker::TrackReferenceKeyFrame()
{
    // 上一帧速度不适用时，假设当前位姿继承上一帧 Tcw
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    // 利用词袋（BoW）找当前帧与参考关键帧 mpReferenceKF 的特征对应项
    // 通过 PoseOptimization 求解当前位姿
    return true;
}

bool Tracker::TrackLocalMap()
{
    // 搜集共视图（Covisibility Graph）相连的 Local KeyFrames 与 Local MapPoints
    // 将 Local MapPoints 投影到当前帧筛选并更新匹配，最后再调用一次 BA 优化 (Optimizer::PoseOptimization)
    return true;
}

bool Tracker::NeedNewKeyFrame()
{
    bool bLocalMappingIdle = mpLocalMapper->SetNotStop();

    // 2. 统计当前帧跟踪到的有效地图点 (Inliers)
    int nMinMatches = 15;
    if (mnMatchesInliers < nMinMatches)
        return false; // 跟踪到的点太少，位姿可能不可靠，不建帧

    // 3. 计算参考关键帧中被跟踪到的地图点数量
    int nRefMatches = 0;
    if (mpReferenceKF)
    {
        // 统计参考关键帧中有效的地图点总数
        std::vector<MapPoint *> vpRefMPs = mpReferenceKF->GetMapPointMatches();
        for (size_t i = 0; i < vpRefMPs.size(); i++)
        {
            if (vpRefMPs[i] && !vpRefMPs[i]->isBad())
                nRefMatches++;
        }
    }

    // 4. 判断时间/帧数间隔条件
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + 20; // 距离上一关键帧已过去 20 帧以上 (强制插入)
    const bool c1b = mCurrentFrame.mnId >= mnLastKeyFrameId + 2;  // 至少间隔 2 帧以上 (防止过度密集)

    // 5. 判断视角变化/地图点重复度条件
    // 如果当前帧跟踪到的地图点数低于参考关键帧的 90%，说明观测到了较多新场景，需要插入关键帧
    bool c2 = false;
    if (nRefMatches > 0)
    {
        float ratioMatches = static_cast<float>(mnMatchesInliers) / static_cast<float>(nRefMatches);
        c2 = ratioMatches < 0.90f;
    }

    // 6. 双目/RGBD 特有逻辑：统计当前帧中的近点 (Close MapPoints) 数量
    // 即使共视比例较高，若新增了足够多的视觉近点，也需要及时插入以提供好的三角化基线
    int nNonTrackedClose = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mCurrentFrame.mThDepth)
        {
            if (!mCurrentFrame.mvpMapPoints[i] || mCurrentFrame.mvbOutlier[i])
                nNonTrackedClose++;
        }
    }
    bool c3 = (nNonTrackedClose > 100); // 发现大量未被跟踪的新近点

    // 7. 综合决策逻辑
    if ((c1a || (c1b && c2) || c3) && bLocalMappingIdle)
    {
        return true;
    }

    return false;
}

void Tracker::CreateNewKeyFrame()
{
    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap.get());
    mpMap->AddKeyFrame(pKF);
    mpReferenceKF = pKF;
    mnLastKeyFrameId = mCurrentFrame.mnId;
}

void Tracker::Reset()
{
    mState = NOT_INITIALIZED;
    mVelocity.setIdentity();
    mpReferenceKF = nullptr;
}
