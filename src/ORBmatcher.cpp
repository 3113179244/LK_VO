#include "ORBmatcher.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : mfNNratio(nnratio), mbCheckOrientation(checkOri) {}

// 计算两个 256 位 ORB 描述子的 Bitwise 汉明距离
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for(int i = 0; i < 8; i++, pa++, pb++) {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

// 核心匹配算法：地图点投影到 Frame 寻找匹配
int ORBmatcher::SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches = 0;
    const bool bFactor = th != 1.0f;

    for(size_t iMP = 0; iMP < vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];

        if(!pMP || pMP->isBad()) continue;

        // 1. 将地图点 3D 世界坐标投影到当前相机坐标系
        Eigen::Vector3f P = pMP->GetWorldPos();
        Eigen::Vector3f Pc = F.GetRotationInverse() * (P - F.GetCameraCenter()); // 利用已有的 Eigen 接口[cite: 3]

        if(Pc.z() <= 0.0f) continue; // 必须在相机前方

        // 2. 投影到像素平面
        const float invz = 1.0f / Pc.z();
        const float u = F.fx * Pc.x() * invz + F.cx; // 使用 Frame 的内参
        const float v = F.fy * Pc.y() * invz + F.cy;

        if(u < F.mnMinX || u > F.mnMaxX || v < F.mnMinY || v > F.mnMaxY)
            continue;

        // 3. 计算半径范围并在网格中寻找候选特征点
        float radius = th * (1.0f + (Pc.z() / pMP->mfMaxDistance)); // 视距越大半径越大
        std::vector<size_t> vIndices = F.GetFeaturesInArea(u, v, radius); // 利用 Frame 提供的 Grid 搜索[cite: 3]

        if(vIndices.empty()) continue;

        // 4. 计算地图点描述子与搜索候选点描述子的汉明距离
        cv::Mat MPdesc = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestLevel = -1;
        int bestDist2 = INT_MAX;
        int bestIdx = -1;

        for(auto idx : vIndices)
        {
            if(F.mvpMapPoints[idx]) continue; // 该点已被匹配则跳过[cite: 3, 4]

            const cv::Mat &d = F.mDescriptors.row(idx);
            int dist = DescriptorDistance(MPdesc, d);

            if(dist < bestDist) {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx = idx;
            } else if(dist < bestDist2) {
                bestDist2 = dist;
            }
        }

        // 5. 严格的比例阈值检查 (Nearest Neighbor Ratio Test)
        if(bestDist <= TH_LOW) {
            if(static_cast<float>(bestDist) < mfNNratio * static_cast<float>(bestDist2)) {
                F.mvpMapPoints[bestIdx] = pMP; // 成功匹配，更新绑定关系[cite: 3, 4]
                nmatches++;
            }
        }
    }

    return nmatches;
}

// 直方图方向筛选：剔除旋转主方向不一致的错配点
void ORBmatcher::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &idx1, int &idx2, int &idx3)
{
    int max1 = 0, max2 = 0, max3 = 0;
    for(int i = 0; i < L; i++) {
        const int s = histo[i].size();
        if(s > max1) {
            max3 = max2; idx3 = idx2;
            max2 = max1; idx2 = idx1;
            max1 = s; idx1 = i;
        } else if(s > max2) {
            max3 = max2; idx3 = idx2;
            max2 = s; idx2 = i;
        } else if(s > max3) {
            max3 = s; idx3 = i;
        }
    }
}