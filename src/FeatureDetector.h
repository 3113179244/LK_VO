#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <numeric> 
// 前向声明，避免头文件循环引用
class Frame;

/**
 * @brief 特征点检测与追踪类
 * 负责帧间光流追踪、特征点检测、屏蔽掩码生成以及双目匹配与外点剔除
 */
class FeatureDetector
{
public:
    FeatureDetector();

    /**
     * @brief 前端图像处理主接口（图像追踪与特征提取流程控制）
     * @param prevFrame 上一帧指针
     * @param currFrame 当前帧指针
     * @param currImgLeft 当前帧左目图像
     * @param currImgRight 当前帧右目图像（单目时可传入空 Mat）
     */
    void TrackImage(std::shared_ptr<Frame> prevFrame,
                    std::shared_ptr<Frame> currFrame,
                    const cv::Mat &prevImgLeft,
                    const cv::Mat &currImgLeft,
                    const cv::Mat &currImgRight);
    void DrawFeaturesOnImage(const cv::Mat &imgLeft, const cv::Mat &imgRight,
                             const std::vector<cv::KeyPoint> &leftKeys,
                             const std::vector<cv::KeyPoint> &rightKeys,
                             const std::vector<int> &trackCnt,
                             cv::Mat &outDisplay);

private:
    /**
     * @brief 利用 LK 金字塔光流算法计算上一帧左图到当前帧左图的特征追踪
     */
    void TrackPrevLeftToCurrLeft(std::shared_ptr<Frame> prevFrame,
                                 std::shared_ptr<Frame> currFrame,
                                 const cv::Mat &prevImg,
                                 const cv::Mat &currImg);

    /**
     * @brief 根据追踪次数（Track Count）对当前帧的特征点进行降序排序
     * 作用：保证被追踪时间更长的稳定特征点优先被保留和使用
     */
    void SortPointsByTrackCount(std::shared_ptr<Frame> currFrame);

    /**
     * @brief 设置特征点提取掩码（Mask）
     * 在现有特征点周围绘制圆形遮罩，防止新提取的特征点过于密集
     */
    void SetMask(std::shared_ptr<Frame> currFrame, int width, int height);

    /**
     * @brief 在掩码允许的区域内，补充提取新的角点（GoodFeaturesToTrack）
     */
    void DetectNewFeatures(std::shared_ptr<Frame> currFrame, const cv::Mat &currImg);

    /**
     * @brief 左右目图像间的双目光流匹配
     */
    void TrackStereo(std::shared_ptr<Frame> currFrame, const cv::Mat &currImgLeft, const cv::Mat &currImgRight);

    /**
     * @brief 通过基础矩阵 (RANSAC) 过滤左右目之间的双目误匹配点
     */
    void FilterStereoMismatch(std::shared_ptr<Frame> currFrame);

    /**
     * @brief 检查像素点坐标是否位于图像有效边界内（防止光流越界）
     */
    bool inBorder(const cv::Point2f &pt, int cols, int rows);

    /**
     * @brief 利用基础矩阵 RANSAC 剔除前后帧之间的光流误匹配
     * @param prevPts 上一帧的像素点 (已通过 status 和边界检查)
     * @param currPts 当前帧对应的像素点
     * @return 内点索引 (对应输入 vector 的下标)
     */
    std::vector<int> FilterInterFrameMismatch(const std::vector<cv::Point2f> &prevPts,
                                              const std::vector<cv::Point2f> &currPts);

    cv::Mat mask;             //特征点提取时的遮罩掩码矩阵
    int maxFeatures;          //图像中维持的最大特征点数量
    int minFeatureDist;       //特征点之间的最小像素距离
    static int nextFeatureId; //用于给新提取特征点分配的全局递增 ID
};

#endif // FEATURE_DETECTOR_H