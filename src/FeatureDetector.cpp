#include "FeatureDetector.h"
#include "Frame.h"
#include "Config.h"
#include <algorithm>
#include <numeric>

// 静态成员变量初始化：用于生成全局唯一的特征点全局 ID
int FeatureDetector::nextFeatureId = 0;

/**
 * @brief 构造函数：从配置类 Config 中读取前端参数
 */
FeatureDetector::FeatureDetector()
{
    maxFeatures = Config::get<int>("max_cnt");     // 帧内最大特征点数
    minFeatureDist = Config::get<int>("min_dist"); // 特征点间最小距离（像素）
}

/**
 * @brief 图像处理核心流程入口
 */
void FeatureDetector::TrackImage(std::shared_ptr<Frame> prevFrame,
                                 std::shared_ptr<Frame> currFrame,
                                 const cv::Mat &currImgLeft,
                                 const cv::Mat &currImgRight)
{
    if (!currFrame)
        return;

    // 1. 若有上一帧，可在此执行前后帧光流跟踪 (需要传入上一帧左图 prevImg)
    // TrackPrevLeftToCurrLeft(prevFrame, currFrame, prevImg, currImgLeft);

    // 2. 将当前已有特征点按追踪次数（稳定度）降序排序
    SortPointsByTrackCount(currFrame);

    // 3. 在已有特征点四周生成 Mask（避免点过于密集），然后补充提取新角点
    SetMask(currFrame, currImgLeft.cols, currImgLeft.rows);
    DetectNewFeatures(currFrame, currImgLeft);

    // 4. 若传入了右目图像，则进行双目光流匹配，并利用 RANSAC 基础矩阵剔除错配
    if (!currImgRight.empty())
    {
        TrackStereo(currFrame, currImgLeft, currImgRight);
        FilterStereoMismatch(currFrame);
    }
}

/**
 * @brief 帧间左图光流追踪（LK Pyramid Optical Flow）
 */
void FeatureDetector::TrackPrevLeftToCurrLeft(std::shared_ptr<Frame> prevFrame,
                                              std::shared_ptr<Frame> currFrame,
                                              const cv::Mat &prevImg,
                                              const cv::Mat &currImg)
{
    if (!prevFrame || prevFrame->mvleftpixel.empty() || !currFrame)
        return;

    // 提取上一帧中的 2D 像素坐标
    std::vector<cv::Point2f> prevPts, currPts;
    for (const auto &kp : prevFrame->mvleftpixel)
    {
        prevPts.push_back(kp.pt);
    }

    std::vector<uchar> status; // LK 光流跟踪状态标识（1表示成功，0表示失败）
    std::vector<float> err;    // 光流跟踪误差

    // 利用 OpenCV 的金字塔 LK 光流算法追踪上一帧特征点到当前帧
    cv::calcOpticalFlowPyrLK(prevImg, currImg, prevPts, currPts, status, err, cv::Size(21, 21), 3);

    // 清空当前帧原本的特征数据，准备写入追踪成功的点
    currFrame->mvleftpixel.clear();
    currFrame->mvFeatureIds.clear();
    currFrame->mvTrackCnt.clear();

    // 遍历光流结果，筛选成功追踪且在图像有效边界内的特征点
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i] && inBorder(currPts[i], currImg.cols, currImg.rows))
        {
            currFrame->mvleftpixel.push_back(cv::KeyPoint(currPts[i], 1.0f));
            currFrame->mvFeatureIds.push_back(prevFrame->mvFeatureIds[i]); // 继承上一帧的点 ID
            currFrame->mvTrackCnt.push_back(prevFrame->mvTrackCnt[i] + 1); // 连续被追踪次数 + 1
        }
    }
    // 更新当前帧记录的特征点总数
    currFrame->iFeaturePointnums = static_cast<int>(currFrame->mvleftpixel.size());
}

/**
 * @brief 按特征点被连续追踪的次数对 Frame 内的特征点数据降序排列
 */
void FeatureDetector::SortPointsByTrackCount(std::shared_ptr<Frame> currFrame)
{
    if (!currFrame || currFrame->mvleftpixel.empty())
        return;

    size_t N = currFrame->mvleftpixel.size();
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0); // 初始化索引数组 [0, 1, 2, ..., N-1]

    // 根据 Frame::mvTrackCnt 进行降序排列（追踪次数多的排前面）
    std::sort(indices.begin(), indices.end(), [currFrame](size_t a, size_t b)
              { return currFrame->mvTrackCnt[a] > currFrame->mvTrackCnt[b]; });

    // 临时存储排序后的结果
    std::vector<cv::KeyPoint> sortedLeft;
    std::vector<int> sortedIds, sortedCnt;

    sortedLeft.reserve(N);
    sortedIds.reserve(N);
    sortedCnt.reserve(N);

    for (size_t idx : indices)
    {
        sortedLeft.push_back(currFrame->mvleftpixel[idx]);
        sortedIds.push_back(currFrame->mvFeatureIds[idx]);
        sortedCnt.push_back(currFrame->mvTrackCnt[idx]);
    }

    // 使用 std::move 移动赋值，提高性能
    currFrame->mvleftpixel = std::move(sortedLeft);
    currFrame->mvFeatureIds = std::move(sortedIds);
    currFrame->mvTrackCnt = std::move(sortedCnt);
}

/**
 * @brief 设置掩码 Mask：在已有特征点处绘制黑色实心圆，抑制在新特征点检测时靠得太近
 */
void FeatureDetector::SetMask(std::shared_ptr<Frame> currFrame, int width, int height)
{
    // 创建全白 (255) 的掩码图像，代表全图区域均可提取
    mask = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));
    if (!currFrame)
        return;

    // 遍历当前已有的特征点，在其周围画半径为 minFeatureDist 的黑色实心圆 (0)
    for (const auto &kp : currFrame->mvleftpixel)
    {
        if (inBorder(kp.pt, width, height))
        {
            if (mask.at<uchar>(kp.pt) == 255)
            {
                cv::circle(mask, kp.pt, minFeatureDist, 0, -1);
            }
        }
    }
}

/**
 * @brief 检测并补充新的特征角点
 */
void FeatureDetector::DetectNewFeatures(std::shared_ptr<Frame> currFrame, const cv::Mat &currImg)
{
    if (!currFrame)
        return;

    // 计算还需补充提取的特征点数量
    int maxToDetect = maxFeatures - static_cast<int>(currFrame->mvleftpixel.size());
    if (maxToDetect > 0)
    {
        std::vector<cv::Point2f> newPts;
        // 在 mask 允许的区域内提取 Shi-Tomasi 角点
        cv::goodFeaturesToTrack(currImg, newPts, maxToDetect, 0.01, minFeatureDist, mask);

        // 将新提取的点追加写入当前帧
        for (const auto &pt : newPts)
        {
            currFrame->mvleftpixel.push_back(cv::KeyPoint(pt, 1.0f));
            currFrame->mvFeatureIds.push_back(nextFeatureId++); // 分配新的全局 ID
            currFrame->mvTrackCnt.push_back(1);                 // 首次被提取，追踪次数设为 1
        }
        currFrame->iFeaturePointnums = static_cast<int>(currFrame->mvleftpixel.size());
    }
}

/**
 * @brief 双目匹配：利用光流将当前帧左图特征点追踪至右图
 */
void FeatureDetector::TrackStereo(std::shared_ptr<Frame> currFrame, const cv::Mat &currImgLeft, const cv::Mat &currImgRight)
{
    if (!currFrame || currFrame->mvleftpixel.empty())
        return;

    std::vector<cv::Point2f> leftPts, rightPts;
    for (const auto &kp : currFrame->mvleftpixel)
    {
        leftPts.push_back(kp.pt);
    }

    std::vector<uchar> status;
    std::vector<float> err;
    // 从左图光流匹配至右图
    cv::calcOpticalFlowPyrLK(currImgLeft, currImgRight, leftPts, rightPts, status, err, cv::Size(21, 21), 3);

    // 调整右图像素存储数组的大小，与左图一一对应
    currFrame->mvrightpixel.clear();
    currFrame->mvrightpixel.resize(currFrame->mvleftpixel.size());

    for (size_t i = 0; i < status.size(); ++i)
    {
        // 追踪成功且在右图边界内
        if (status[i] && inBorder(rightPts[i], currImgRight.cols, currImgRight.rows))
        {
            currFrame->mvrightpixel[i] = cv::KeyPoint(rightPts[i], 1.0f);
        }
        else
        {
            // 追踪失败或越界，用负坐标标志位 (-1, -1) 表示无效
            currFrame->mvrightpixel[i] = cv::KeyPoint(cv::Point2f(-1.0f, -1.0f), 1.0f);
        }
    }
}

/**
 * @brief 双目外点剔除：利用 RANSAC 计算基础矩阵 (Fundamental Matrix)，标记误匹配点
 */
void FeatureDetector::FilterStereoMismatch(std::shared_ptr<Frame> currFrame)
{
    if (!currFrame)
        return;

    std::vector<cv::Point2f> ptsLeft, ptsRight;
    std::vector<int> validIndices; // 收集有有效右图匹配点的原始索引

    for (size_t i = 0; i < currFrame->mvleftpixel.size(); ++i)
    {
        if (i < currFrame->mvrightpixel.size() && currFrame->mvrightpixel[i].pt.x >= 0)
        {
            ptsLeft.push_back(currFrame->mvleftpixel[i].pt);
            ptsRight.push_back(currFrame->mvrightpixel[i].pt);
            validIndices.push_back(static_cast<int>(i));
        }
    }

    // 匹配点至少达到 8 个，才能求解基础矩阵 (8-point algorithm)
    if (ptsLeft.size() >= 8)
    {
        std::vector<uchar> status;
        // 使用 RANSAC 算法求解基础矩阵，同时判断内点/外点
        cv::findFundamentalMat(ptsLeft, ptsRight, cv::FM_RANSAC, 3.0, 0.99, status);

        for (size_t i = 0; i < status.size(); ++i)
        {
            // 被 RANSAC 标记为外点的匹配对，重置其右图坐标为无效点 (-1, -1)
            if (!status[i])
            {
                int idx = validIndices[i];
                currFrame->mvrightpixel[idx] = cv::KeyPoint(cv::Point2f(-1.0f, -1.0f), 1.0f);
            }
        }
    }
}

/**
 * @brief 判断给定的 2D 像素坐标是否落在图像的合法内部（留出 BORDER_SIZE 边界）
 */
bool FeatureDetector::inBorder(const cv::Point2f &pt, int cols, int rows)
{
    const int BORDER_SIZE = 1; // 边界保护宽度
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);

    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

void FeatureDetector::DrawFeaturesOnImage(const cv::Mat &imgLeft, const cv::Mat &imgRight,
                                          const std::vector<cv::KeyPoint> &leftKeys,
                                          const std::vector<cv::KeyPoint> &rightKeys,
                                          cv::Mat &outDisplay)
{
    cv::Mat colorLeft, colorRight;

    // 转为 3 通道 BGR 彩色图
    if (imgLeft.channels() == 1)
        cv::cvtColor(imgLeft, colorLeft, cv::COLOR_GRAY2BGR);
    else
        colorLeft = imgLeft.clone();

    if (imgRight.channels() == 1)
        cv::cvtColor(imgRight, colorRight, cv::COLOR_GRAY2BGR);
    else
        colorRight = imgRight.clone();

    // 绘制左图特征点（绿点）
    for (const auto &kp : leftKeys)
    {
        cv::circle(colorLeft, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
    }

    // 绘制右图特征点（黄点）
    for (const auto &kp : rightKeys)
    {
        if (kp.pt.x >= 0 && kp.pt.y >= 0) // 过滤掉无效的点 (-1, -1)
        {
            cv::circle(colorRight, kp.pt, 3, cv::Scalar(0, 255, 255), -1);
        }
    }

    // 上下拼接并输出
    cv::vconcat(colorLeft, colorRight, outDisplay);
}