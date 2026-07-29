#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Camera.h"

// 初始化静态成员（分配唯一ID）
long unsigned int KeyFrame::nNextId = 0;

/**
 * @brief 构造函数：从普通帧拷贝关键数据
 * @param pFrame 普通帧指针
 */
KeyFrame::KeyFrame(std::shared_ptr<Frame> pFrame)
    : mId(nNextId++),                      // 生成全局唯一ID
      mFrameId(pFrame->mFrameId),          // 拷贝原始帧ID
      mTimeStamp(pFrame->mTimeStamp),      // 拷贝时间戳
      N(pFrame->iFeaturePointnums),        // 拷贝特征点数量
      mvKeysLeft(pFrame->mvleftpixel),     // 拷贝特征点（像素坐标）
      mvpMapPoints(pFrame->mvpMapPoints)   // 拷贝地图点指针列表（浅拷贝）
{
    mpCamera = nullptr;                     // 相机指针暂为空，后续由外部赋值
    SetPose(pFrame->GetPose());            // 拷贝位姿
}

/**
 * @brief 设置位姿，同时计算光心位置
 * @param Tcw 相机到世界的变换矩阵（4x4）
 */
void KeyFrame::SetPose(const Eigen::Matrix4f &Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    mTcw = Tcw;
    // 提取旋转矩阵Rcw和平移向量tcw
    Eigen::Matrix3f Rcw = mTcw.block<3, 3>(0, 0);
    Eigen::Vector3f tcw = mTcw.block<3, 1>(0, 3);
    // 光心坐标 Ow = -Rcw^T * tcw （因为 Tcw 表示从世界到相机，其逆为 Twc = [R^T, -R^T*t]）
    mOw = -Rcw.transpose() * tcw;
}

/**
 * @brief 获取位姿（线程安全）
 */
Eigen::Matrix4f KeyFrame::GetPose()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mTcw;
}

/**
 * @brief 获取相机光心坐标（线程安全）
 */
Eigen::Vector3f KeyFrame::GetCameraCenter()
{
    std::unique_lock<std::mutex> lock(mMutexPose);
    return mOw;
}

/**
 * @brief 获取所有地图点列表（线程安全，返回副本）
 */
std::vector<std::shared_ptr<MapPoint>> KeyFrame::GetMapPoints()
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints;  // 返回副本，但内部指针仍指向原对象
}

/**
 * @brief 获取指定索引的地图点（线程安全）
 */
std::shared_ptr<MapPoint> KeyFrame::GetMapPoint(const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

/**
 * @brief 在指定索引处设置地图点
 */
void KeyFrame::AddMapPoint(std::shared_ptr<MapPoint> pMP, const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

/**
 * @brief 删除指定索引的地图点（设为nullptr）
 */
void KeyFrame::EraseMapPointMatch(const size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

/**
 * @brief 删除所有指向某地图点的匹配（用于地图点被删除时清理）
 */
void KeyFrame::EraseMapPointMatch(std::shared_ptr<MapPoint> pMP)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);
    for (size_t i = 0; i < mvpMapPoints.size(); i++)
    {
        if (mvpMapPoints[i] == pMP)
        {
            mvpMapPoints[i] = nullptr;
        }
    }
}

/**
 * @brief 增加或更新与另一个关键帧的共视权重
 */
void KeyFrame::AddConnection(std::shared_ptr<KeyFrame> pKF, const int weight)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (!mConnectedKeyFrameWeights.count(pKF))
        mConnectedKeyFrameWeights[pKF] = weight;
    else
        mConnectedKeyFrameWeights[pKF] += weight;  // 累加权重
}

/**
 * @brief 移除与某个关键帧的共视关系
 */
void KeyFrame::EraseConnection(std::shared_ptr<KeyFrame> pKF)
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    if (mConnectedKeyFrameWeights.count(pKF))
        mConnectedKeyFrameWeights.erase(pKF);
}

/**
 * @brief 获取所有共视关键帧及权重（返回副本）
 */
std::map<std::shared_ptr<KeyFrame>, int> KeyFrame::GetConnectedKeyFrames()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);
    return mConnectedKeyFrameWeights;
}