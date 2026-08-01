#include "Frame.h"
#include <thread>
#include <cmath>

// 静态变量初始化
long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::fx = 0, Frame::fy = 0, Frame::cx = 0, Frame::cy = 0, Frame::invfx = 0, Frame::invfy = 0;
float Frame::mnMinX = 0, Frame::mnMinY = 0, Frame::mnMaxX = 0, Frame::mnMaxY = 0;
float Frame::mfGridElementWidthInv = 0, Frame::mfGridElementHeightInv = 0;

Frame::Frame() {}

Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, 
             ORBextractor* extractorLeft, ORBextractor* extractorRight, 
             ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    : mTimeStamp(timeStamp), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight), 
      mpORBvocabulary(voc), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // 分配帧 ID
    mnId = nNextId++;

    // 1. 多线程并发提取左右图 ORB 特征
    std::thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
    std::thread threadRight(&Frame::ExtractORB, this, 1, imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    // (此处应包含 UndistortKeyPoints，若输入已极线校正可直接拷贝 mvKeysUn = mvKeys)
    mvKeysUn = mvKeys; 

    // 2. 双目匹配计算深度
    ComputeStereoMatches();

    // 3. 初始化地图点和外点标记数组
    mvpMapPoints = std::vector<MapPoint*>(N, nullptr);    
    mvbOutlier = std::vector<bool>(N, false);

    // 4. 初始化图像边界和相机内参 (仅计算一次)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    mb = mbf / fx; // 物理基线长度

    // 5. 将特征点划分到网格
    AssignFeaturesToGrid();
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    // 实际项目中提取器需要重载 operator() 来输出特征点和描述子
    // 如果 flag == 0 则处理左图，否则处理右图
}

void Frame::SetPose(const Eigen::Matrix4f &Tcw)
{
    mTcw = Tcw;
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.block<3,3>(0,0);
    mRwc = mRcw.transpose();
    mtcw = mTcw.block<3,1>(0,3);
    mOw = -mRwc * mtcw; // 计算相机光心在世界坐标系下的 3D 坐标
}

void Frame::ComputeStereoMatches()
{
    mvuRight = std::vector<float>(N, -1.0f);
    mvDepth = std::vector<float>(N, -1.0f);

    // 此处为简化版框架：
    // 在真实实现中，你需要遍历左图特征点 mvKeys[i]，
    // 通过极线约束，在 mvKeysRight 中寻找 y 坐标相近的特征点，计算描述子距离，
    // 找到匹配点后，求取视差 disparity = left_x - right_x
    // 最后深度 mvDepth[i] = mbf / disparity;
}

Eigen::Vector3f Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z > 0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        
        // 像素坐标转相机坐标
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        
        // 相机坐标转世界坐标: P_world = Rwc * P_camera + Ow
        return mRwc * x3Dc + mOw;
    }
    return Eigen::Vector3f::Zero();
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果图像已去畸变，边界即为图像本身大小
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS; i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    if(posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if(nMinCellX >= FRAME_GRID_COLS) return vIndices;

    const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if(nMaxCellX < 0) return vIndices;

    const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if(nMinCellY >= FRAME_GRID_ROWS) return vIndices;

    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if(nMaxCellY < 0) return vIndices;

    // 遍历对应网格区域提取特征点索引
    for(int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty()) continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                
                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                if(fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}