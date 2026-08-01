#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <vector>
#include <opencv2/opencv.hpp>

class Frame; // 前置声明 Frame 帧类

class ORBmatcher
{
public:
    /**
     * @brief 构造函数
     * @param nnratio            最优与次优距离的比值阈值（用于剔除模糊匹配，通常取 0.6~0.8）
     * @param checkOrientation   是否检查旋转一致性（直方图筛选）
     */
    ORBmatcher(float nnratio = 0.6f, bool checkOrientation = true);
    ~ORBmatcher() = default;

    /**
     * @brief 核心计算工具：计算两个 ORB 描述子的汉明距离 (Hamming Distance)
     * 利用 CPU 硬件指令 popcount 进行高性能加速
     */
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    /**
     * @brief 双目极线搜索匹配：在左右目图像中匹配特征点，并直接计算视差 (Disparity) 和深度 (Depth)
     * @param F 包含左右图像特征信息的 Frame 引用
     * @return 成功建立双目匹配的数量
     */
    int ComputeStereoMatches(Frame &F);

    static const int TH_LOW;       // 匹配距离较低阈值
    static const int TH_HIGH;      // 匹配距离较高阈值
    static const int HISTO_LENGTH; // 方向直方图的 Bin 数量 (36 个 bin，每 10 度一个)

private:
    // 计算旋转直方图中前三个最大值的索引（用于剔除不一致的主方向）
    void ComputeThreeBestIdx(int* histo, const int L, int &idx1, int &idx2, int &idx3);

    float mfNNratio;            // 最优/次优距离比率
    bool mbCheckOrientation;    // 角度检查标志位
};

#endif // ORBMATCHER_H