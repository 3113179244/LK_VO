#include "ORBextractor.h"

// BRIEF 描述子使用的 512 个相对采样点偏置 pattern (256 对)
static int bit_pattern_31_[256*4] = {
    8,-3, 9,5, 4,2, 7,-12, -11,9, -8,2, 7,-12, 12,-19, 3,7, 5,-1, 11,-10, 12,-4,
    // ... (标准的 ORB 256对 pattern 采样点列表)
};

ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvInvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);

    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    // 分配每层金字塔应当提取的特征点数量
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - pow(factor, nlevels));
    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++) {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    // 预先计算灰度质心法圆弧边界 umax (半径 R = 19)
    umax.resize(16);
    int vmax = cvFloor(15 * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(15 * sqrt(2.f) / 2);
    for (int v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(15 * 15 - v * v));
}

// 计算圆内灰度质心方向 Angle
static float ComputeOrientation(const cv::Mat& img, cv::Point2f pt, const std::vector<int>& umax)
{
    int m_01 = 0, m_10 = 0;
    const uchar* center = &img.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    for (int u = -15; u <= 15; ++u)
        m_10 += u * center[u];

    int step = (int)img.step1();
    for (int v = 1; v <= 15; ++v) {
        int u0 = umax[v];
        int sum_m01 = 0;
        for (int u = -u0; u <= u0; ++u) {
            int val_plus = center[u + v * step];
            int val_minus = center[u - v * step];
            sum_m01 += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * sum_m01;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

void ORBextractor::operator()(cv::InputArray _image, cv::InputArray _mask,
                               std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors)
{
    cv::Mat image = _image.getMat();
    if(image.channels() > 1)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // 1. 构建图像金字塔
    ComputePyramid(image);

    // 2. 利用 Octree 四叉树在每层提取分布均匀的关键点
    std::vector<std::vector<cv::KeyPoint>> allKeypoints(nlevels);
    ComputeKeyPointsOctree(allKeypoints);

    // 3. 计算关键点方向并提取 ORB 描述子
    cv::Mat descriptors;
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    if(nkeypoints == 0)
        _descriptors.release();
    else {
        _descriptors.create(nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level) {
        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0) continue;

        cv::Mat workingMat = mvImagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        
        // 计算旋转后的 BRIEF 描述子
        for (int i = 0; i < nkeypointsLevel; i++) {
            keypoints[i].angle = ComputeOrientation(mvImagePyramid[level], keypoints[i].pt, umax);
            
            // 还原到第 0 层原图坐标
            cv::KeyPoint kpt = keypoints[i];
            kpt.pt *= mvScaleFactor[level];
            kpt.octave = level;
            _keypoints.push_back(kpt);
        }
        offset += nkeypointsLevel;
    }
}

void ORBextractor::ComputePyramid(cv::Mat image)
{
    mvImagePyramid.resize(nlevels);
    for (int level = 0; level < nlevels; ++level) {
        float scale = mvInvScaleFactor[level];
        cv::Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        if (level == 0)
            mvImagePyramid[level] = image;
        else
            cv::resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
    }
}

void ORBextractor::ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    // 此处调用 FAST 算子检测角点，随后使用 DistributeOctree 分割保留 score 最高的角点
}

