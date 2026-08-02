#include "Frame.h"
#include "ORBextractor.h"
#include "ORBmatcher.h"
#include <thread>
#include <cmath>
#include "Config.h"
// 静态变量初始化
long unsigned int Frame::nNextId = 0;
bool Frame::mbInitialComputations = true;
float Frame::fx = 0, Frame::fy = 0, Frame::cx = 0, Frame::cy = 0, Frame::invfx = 0, Frame::invfy = 0;
float Frame::mnMinX = 0, Frame::mnMinY = 0, Frame::mnMaxX = 0, Frame::mnMaxY = 0;
float Frame::mfGridElementWidthInv = 0, Frame::mfGridElementHeightInv = 0;

Frame::Frame() {}

// 双目帧构造函数实现
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp,
             ORBextractor *extractorLeft, ORBextractor *extractorRight,
             ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    : mTimeStamp(timeStamp), mpORBextractorLeft(extractorLeft), mpORBextractorRight(extractorRight),
      mpORBvocabulary(voc), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // 分配唯一的帧 ID，并递增计数器
    mnId = nNextId++;

    // 多线程并发提取左右图 ORB 特征，以加速前端处理
    std::thread threadLeft(&Frame::ExtractORB, this, 0, imLeft);
    std::thread threadRight(&Frame::ExtractORB, this, 1, imRight);
    threadLeft.join();
    threadRight.join();

    // 记录特征点总数
    N = mvKeys.size();
    if (mvKeys.empty())
        return;

    // 此处简化了去畸变过程。在实际系统中，若输入图像已极线校正，可直接拷贝 mvKeysUn = mvKeys
    mvKeysUn = mvKeys;

    // 双目匹配，通过左右目特征点匹配计算视差，进而获得深度信息
    ComputeStereoMatches();

    // 初始化地图点和外点标记数组，大小为特征点总数 N
    mvpMapPoints = std::vector<MapPoint *>(N, nullptr);
    mvbOutlier = std::vector<bool>(N, false);

    // 初始化图像边界和相机内参。因为是静态变量，只需在程序启动时(第一帧)计算一次
    if (mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        // 计算网格宽度和高度的倒数，用于后续将坐标快速映射到网格索引 (乘法比除法快)
        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / (mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / (mnMaxY - mnMinY);

        // 提取相机内参矩阵中的参数
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

    // 计算物理基线长度 $b = \frac{bf}{f_x}$
    mb = mbf / fx;
    mImGrayLeft = imLeft.clone();
    mImGrayRight = imRight.clone(); 
    // 将特征点划分到网格中，加速局部区域特征匹配搜索
    AssignFeaturesToGrid();
}

// 提取 ORB 特征
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if (flag == 0)
    {
        // 左图：使用左目提取器，提取特征点存入 mvKeys，描述子存入 mDescriptors
        if (mpORBextractorLeft)
        {
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        }
    }
    else
    {
        // 右图：使用右目提取器，结果存入 mvKeysRight 和 mDescriptorsRight
        if (mpORBextractorRight)
        {
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
        }
    }
}

// 设置相机位姿矩阵
void Frame::SetPose(const Eigen::Matrix4f &Tcw)
{
    mTcw = Tcw;
    UpdatePoseMatrices(); // 位姿更新后，同步更新其分解的各个矩阵和向量
}

// 依据最新的 $T_{cw}$ 更新旋转平移矩阵
void Frame::UpdatePoseMatrices()
{
    // 从 4x4 变换矩阵中提取 3x3 旋转矩阵 $R_{cw}$
    mRcw = mTcw.block<3, 3>(0, 0);
    // 计算其转置 (即逆矩阵) $R_{wc}$
    mRwc = mRcw.transpose();
    // 从 4x4 变换矩阵中提取 3x1 平移向量 $t_{cw}$
    mtcw = mTcw.block<3, 1>(0, 3);
    // 计算相机光心在世界坐标系下的 3D 坐标: $O_w = -R_{cw}^T \cdot t_{cw} = -R_{wc} \cdot t_{cw}$
    mOw = -mRwc * mtcw;
}

// 计算双目匹配以获取深度 (此处为框架示意代码)
void Frame::ComputeStereoMatches()
{
    // 初始化容器：N 为左图提取的特征点总数
    mvuRight = std::vector<float>(N, -1.0f);
    mvDepth = std::vector<float>(N, -1.0f);

    // 参数设置
    const float thOrbDist = 60.0f; // ORB 描述子汉明距离阈值
    const float thY = 1.5f;        // 极线约束：y 方向允许的最大像素偏差
    const float minZ = 0.0f;       // 最小有效深度
    const float minD = 0.0f;       // 最小视差（必须 > 0）
    const float maxD = fx;         // 最大视差，可设为 fx 或图像宽度，防止除零

    // 为右图特征点建立 y 坐标索引，避免每次全量遍历
    const int nRows = Config::g_nImageHeight; // 可根据 Config::g_nImageHeight 动态设置
    std::vector<std::vector<size_t>> vRowIndices(nRows);
    for (size_t iR = 0; iR < mvKeysRight.size(); iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &y = kp.pt.y;
        const int row = cvRound(y);
        if (row >= 0 && row < nRows)
            vRowIndices[row].push_back(iR);
    }

    // 遍历左图每个特征点，在右图寻找最佳匹配
    for (int iL = 0; iL < N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const float &uL = kpL.pt.x;
        const float &vL = kpL.pt.y;
        const int levelL = kpL.octave;
        const float &vL_rounded = vL;

        // 只考虑第 0 层（或根据需求放宽）的特征点进行双目匹配，
        // 因为高金字塔层分辨率低，视差计算误差大。
        // 如果追求速度可保留此限制；若追求点数量可注释掉。
        // if (levelL != 0) continue;

        // 在右图中收集 y 坐标相近的候选点
        std::vector<size_t> vCandidates;
        const int minRow = std::max(0, (int)(vL_rounded - thY));
        const int maxRow = std::min(nRows - 1, (int)(vL_rounded + thY));

        for (int row = minRow; row <= maxRow; row++)
        {
            for (size_t idx : vRowIndices[row])
            {
                const cv::KeyPoint &kpR = mvKeysRight[idx];
                // 金字塔层级差异不能太大（保证尺度一致性）
                if (abs(kpR.octave - levelL) > 1)
                    continue;
                vCandidates.push_back(idx);
            }
        }

        if (vCandidates.empty())
            continue;

        // 计算描述子距离，寻找最佳和次佳匹配
        const cv::Mat &dL = mDescriptors.row(iL);
        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX; // 次佳距离，用于 Ratio Test
        int bestIdxR = -1;

        for (size_t iRc : vCandidates)
        {
            const cv::Mat &dR = mDescriptorsRight.row(iRc);
            const int dist = ORBmatcher::DescriptorDistance(dL, dR);

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdxR = iRc;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        // 匹配质量过滤
        // 最佳距离必须小于阈值
        if (bestDist > thOrbDist)
            continue;
        // 唯一性比率测试：最佳匹配必须明显优于次佳匹配（避免模糊匹配）
        if (bestDist2 > 0 && static_cast<float>(bestDist) > 0.8f * static_cast<float>(bestDist2))
            continue;

        // 计算视差与深度
        const float uR = mvKeysRight[bestIdxR].pt.x;
        const float disparity = uL - uR; // 视差：左 x - 右 x（必须为正）

        // 视差有效性检查
        if (disparity <= minD || disparity >= maxD)
            continue;

        // 深度计算：Z = (baseline * fx) / disparity
        const float depth = mbf / disparity;

        // 深度有效性检查
        if (depth <= minZ || depth > mThDepth)
            continue;

        // 保存结果
        mvuRight[iL] = uR;
        mvDepth[iL] = depth;
    }
}

// 将指定特征点反投影为 3D 世界坐标
Eigen::Vector3f Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if (z > 0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;

        // 像素坐标转相机坐标：
        // $X_c = \frac{(u - c_x) \cdot Z}{f_x}$
        // $Y_c = \frac{(v - c_y) \cdot Z}{f_y}$
        const float x = (u - cx) * z * invfx;
        const float y = (v - cy) * z * invfy;
        Eigen::Vector3f x3Dc(x, y, z);

        // 相机坐标转世界坐标: $P_{world} = R_{wc} \cdot P_{camera} + O_w$
        return mRwc * x3Dc + mOw;
    }
    return Eigen::Vector3f::Zero(); // 如果没有有效的深度值，则返回全零向量
}

// 获取图像有效区域边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 在这里假设图像已去畸变或不需要考虑畸变引起的边界收缩问题，边界即为图像本身尺寸
    mnMinX = 0.0f;
    mnMaxX = imLeft.cols;
    mnMinY = 0.0f;
    mnMaxY = imLeft.rows;
}

// 将图像中的特征点分配到离散网格中
void Frame::AssignFeaturesToGrid()
{
    // 预分配每个网格点的内存空间以避免动态扩容带来的时间开销
    // 假设特征点是均匀分布的，每个网格大约会有 0.5 * N / (Rows * Cols) 个点
    int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
    for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
        for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
            mGrid[i][j].reserve(nReserve);

    // 遍历所有的去畸变特征点，计算其对应的网格坐标并存入
    for (int i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];
        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i); // 将该特征点的索引加入对应网格
    }
}

// 根据特征点像素坐标，计算对应的网格索引，判断是否在图像网格范围内
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

    // 检查是否越界
    if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

// 快速查找指定区域内所有的特征点索引 (主要用于特征追踪阶段)
std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel, const int maxLevel) const
{
    std::vector<size_t> vIndices;
    vIndices.reserve(N);

    // 计算包含搜索圆形的最小和最大网格索引，限定在合法范围内
    const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    // 双层循环：仅遍历落入上述边界范围内的网格区域
    for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
        {
            const std::vector<size_t> vCell = mGrid[ix][iy]; // 取出该网格内的所有特征点索引
            if (vCell.empty())
                continue;

            for (size_t j = 0, jend = vCell.size(); j < jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];

                const float distx = kpUn.pt.x - x;
                const float disty = kpUn.pt.y - y;

                // 判断欧氏距离的近似条件：通过方盒模型初步筛选落入搜索半径 r 的点
                if (fabs(distx) < r && fabs(disty) < r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }
    return vIndices;
}