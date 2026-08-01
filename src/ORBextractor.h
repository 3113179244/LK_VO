#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

// 四叉树节点，用于特征点均匀化分布
class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2f UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORBextractor
{
public:
    enum { IS_FAST_SCORE = 0, IS_HARRIS_SCORE = 1 };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    ~ORBextractor() = default;

    // 核心重载运算符：输入图像，输出关键点与描述子
    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    // 参数 Getter 接口
    inline int GetLevels() const { return nlevels; }
    inline float GetScaleFactor() const { return scaleFactor; }
    inline std::vector<float> GetScaleFactors() const { return mvScaleFactor; }
    inline std::vector<float> GetScaleSigmaSquares() const { return mvLevelSigma2; }
    inline std::vector<float> GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }

public:
    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;
    std::vector<int> umax;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

    std::vector<cv::Mat> mvImagePyramid;

protected:
    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                               const float &minX, const float &maxX,
                                               const float &minY, const float &maxY,
                                               const int &nFeatures, const int &level);
};

#endif // ORBEXTRACTOR_H