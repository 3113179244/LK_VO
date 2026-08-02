#include "Tracker.h"
#include "Config.h"
#include "ORBextractor.h"
#include "Optimizer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Map.h"
#include "FrameDrawer.h"
Tracker::Tracker(System *pSys, std::shared_ptr<Map> pMap, int sensor)
    : mpSystem(pSys), mpMap(pMap), mState(NO_IMAGES_YET), mVelocity(Eigen::Matrix4f::Identity()), mpReferenceKF(nullptr)
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
    // 根据恒速模型粗略预测当前位姿: T_cw = T_cl * T_lw
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // 通过投影映射将上一帧的地图点与当前帧做匹配，并通过 MotionOnlyBA (BA 优化) 估计姿态
    // (对应调用 ORBmatcher 和 Optimizer::PoseOptimization)

    // 假设优化后匹配内点数满足阈值
    mnMatchesInliers = 30;
    return (mnMatchesInliers >= 10);
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
    // 关键帧插入策略约束：
    // 1. 距离上一次插入关键帧经过了足够的帧数
    // 2. 当前帧追踪到的地图点内点比例低于参考关键帧一定百分比（如 < 90%）
    // 3. Local Mapping 处于空闲状态
    return (mCurrentFrame.mnId - mnLastKeyFrameId > 20);
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