#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

// 四叉树节点：用于特征点在空间上的均匀分布
class ExtractorNode
{
public:
    ExtractorNode() : bNoMore(false) {}
    // 将当前节点一分为四（四个子节点）
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys; // 当前节点包含的特征点
    cv::Point2f UL, UR, BL, BR;      // 节点的四个顶点坐标（左上、右上、左下、右下）
    std::list<ExtractorNode>::iterator lit; // 在链表中的迭代器位置
    bool bNoMore;                    // 标记该节点是否不可再分割（例如仅剩1个特征点）
};

class ORBextractor
{
public:
    enum { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    /**
     * @brief 构造函数：初始化 ORB 提取器参数与金字塔层级信息
     * @param nfeatures     期望提取的特征点总数
     * @param scaleFactor   金字塔缩放因子（通常取 1.2）
     * @param nlevels       金字塔层数（通常取 8）
     * @param iniThFAST     初始 FAST 角点检测阈值（较严格）
     * @param minThFAST     最小 FAST 角点检测阈值（较宽松，初始阈值找不到时使用）
     */
    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    ~ORBextractor() = default;

    /**
     * @brief 重载 () 运算符（仿函数）：完成特征提取的核心接口
     * @param image        输入图像
     * @param mask         掩码矩阵
     * @param keypoints    输出提取到的关键点集合
     * @param descriptors  输出计算得到的 ORB 描述子矩阵 (N x 32 uchar)
     */
    void operator()(cv::InputArray image, cv::InputArray mask,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors);

    // Getters 接口
    int GetLevels() const { return nlevels; }
    float GetScaleFactor() const { return scaleFactor; }
    const std::vector<float>& GetScaleFactors() const { return mvScaleFactor; }
    const std::vector<float>& GetInverseScaleFactors() const { return mvInvScaleFactor; }
    const std::vector<float>& GetScaleSigmaSquares() const { return mvLevelSigma2; }
    const std::vector<float>& GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }

private:
    void ComputePyramid(const cv::Mat& image); // 构建图像金字塔
    void ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints); // 基于四叉树提取角点
    
    // 使用四叉树分割算法，使特征点均匀分布
    std::vector<cv::KeyPoint> DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                               const int &minX, const int &maxX,
                                               const int &minY, const int &maxY,
                                               const int &nFeatures, const int &level);

    // 使用灰度质心法计算特征点的主方向（实现旋转不变性）
    void ComputeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, const std::vector<int>& umax);

    int nfeatures;        // 提取总特征点数
    float scaleFactor;    // 金字塔层间缩放比例
    int nlevels;          // 金字塔总层数
    int iniThFAST;        // 初始 FAST 阈值
    int minThFAST;        // 备用 FAST 阈值

    std::vector<cv::Mat> mvImagePyramid;       // 存储每层金字塔图像
    std::vector<int> mnFeaturesPerLevel;       // 每层金字塔分配提取的特征点数量
    std::vector<float> mvScaleFactor;          // 每层的累计缩放因子
    std::vector<float> mvInvScaleFactor;       // 缩放因子的倒数
    std::vector<float> mvLevelSigma2;          // 尺度方差 (scale^2)
    std::vector<float> mvInvLevelSigma2;       // 尺度方差倒数

    std::vector<cv::Point> pattern;            // BRIEF 描述子的采样点对模式 (256对)
    std::vector<int> umax;                     // 灰度质心计算圆域每行的最大 x 偏移量
};

#endif // ORBEXTRACTOR_H