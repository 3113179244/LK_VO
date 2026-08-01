#include "ORBmatcher.h"
#include "Frame.h"
#include <limits>
#include "ORBextractor.h"
#include <algorithm>
const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 36;

ORBmatcher::ORBmatcher(float nnratio, bool checkOrientation)
    : mfNNratio(nnratio), mbCheckOrientation(checkOrientation) {}

int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;
    for (int i = 0; i < 8; i++) {
        unsigned int int_or = pa[i] ^ pb[i];
        dist += __builtin_popcount(int_or); // 使用 GCC 内置 POPCNT 硬件加速
    }
    return dist;
}

int ORBmatcher::ComputeStereoMatches(Frame &F)
{
    int nMatches = 0;
    F.mvuRight = std::vector<float>(F.N, -1.0f);
    F.mvDepth = std::vector<float>(F.N, -1.0f);

    const int nRows = F.mbInitialComputations ? 480 : F.mnMaxY; // 图像行数基准
    std::vector<std::vector<size_t>> vRowIndices(nRows, std::vector<size_t>());

    for (int i = 0; i < nRows; i++)
        vRowIndices[i].reserve(10);

    const int Nr = F.mvKeysRight.size();
    for (int iR = 0; iR < Nr; iR++) {
        const cv::KeyPoint &kp = F.mvKeysRight[iR];
        const float kpY = kp.pt.y;
        const float r = 2.0f * F.mpORBextractorRight->GetScaleFactors()[kp.octave];
        const int maxr = ceil(kpY + r);
        const int minr = floor(kpY - r);

        for (int yi = minr; yi <= maxr; yi++) {
            if (yi >= 0 && yi < nRows)
                vRowIndices[yi].push_back(iR);
        }
    }

    const float minZ = F.mb;
    const float minD = 0;
    const float maxD = F.mbf / minZ;

    std::vector<std::pair<int, int>> vDistIdx;
    vDistIdx.reserve(F.N);

    std::vector<int> rotHistogram[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; i++)
        rotHistogram[i].reserve(F.N);

    const float factor = 1.0f / HISTO_LENGTH;

    for (int iL = 0; iL < F.N; iL++) {
        const cv::KeyPoint &kpL = F.mvKeys[iL];
        const int levelL = kpL.octave;
        const float vL = kpL.pt.y;
        const float uL = kpL.pt.x;

        const std::vector<size_t> &vCandidates = vRowIndices[(int)vL];
        if (vCandidates.empty()) continue;

        const float minU = uL - maxD;
        const float maxU = uL - minD;
        if (maxU < 0) continue;

        int bestDist = TH_HIGH;
        size_t bestIdxR = 0;
        const cv::Mat &dL = F.mDescriptors.row(iL);

        for (size_t iC = 0; iC < vCandidates.size(); iC++) {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = F.mvKeysRight[iR];

            if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1) continue;

            const float uR = kpR.pt.x;
            if (uR >= minU && uR <= maxU) {
                const cv::Mat &dR = F.mDescriptorsRight.row(iR);
                const int dist = DescriptorDistance(dL, dR);

                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        if (bestDist < TH_HIGH) {
            const cv::KeyPoint &kpR = F.mvKeysRight[bestIdxR];
            const float uR = kpR.pt.x;
            const float disparity = uL - uR;

            if (disparity >= 0) {
                const float depth = F.mbf / disparity;
                F.mvuRight[iL] = uR;
                F.mvDepth[iL] = depth;
                nMatches++;
            }
        }
    }

    return nMatches;
}

void ORBmatcher::ComputeThreeBestIdx(int* histo, const int L, int &idx1, int &idx2, int &idx3)
{
    int max1 = 0, max2 = 0, max3 = 0;
    for (int i = 0; i < L; i++) {
        const int n = histo[i];
        if (n > max1) {
            max3 = max2; max2 = max1; max1 = n;
            idx3 = idx2; idx2 = idx1; idx1 = i;
        } else if (n > max2) {
            max3 = max2; max2 = n;
            idx3 = idx2; idx2 = i;
        } else if (n > max3) {
            max3 = n; idx3 = i;
        }
    }

    if (max2 < 0.1f * max1) {
        idx2 = -1; idx3 = -1;
    } else if (max3 < 0.1f * max1) {
        idx3 = -1;
    }
}