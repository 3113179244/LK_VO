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
                                 const cv::Mat &prevImgLeft,
                                 const cv::Mat &currImgLeft,
                                 const cv::Mat &currImgRight)
{
    if (!currFrame)
        return;

    // 若有上一帧，可在此执行前后帧光流跟踪 (需要传入上一帧左图 prevImg)
    if (prevFrame && !prevImgLeft.empty())
    {
        TrackPrevLeftToCurrLeft(prevFrame, currFrame, prevImgLeft, currImgLeft);
    }

    // 将当前已有特征点按追踪次数（稳定度）降序排序
    SortPointsByTrackCount(currFrame);

    // 在已有特征点四周生成 Mask（避免点过于密集），然后补充提取新角点
    SetMask(currFrame, currImgLeft.cols, currImgLeft.rows);
    DetectNewFeatures(currFrame, currImgLeft);

    // 若传入了右目图像，则进行双目光流匹配，并利用 RANSAC 基础矩阵剔除错配
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

    // 第一步：初步筛选（status 成功且在边界内）
    std::vector<cv::Point2f> validPrevPts, validCurrPts;
    std::vector<int> validIndices; // 对应原始 prevPts 的下标
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i] && inBorder(currPts[i], currImg.cols, currImg.rows))
        {
            validPrevPts.push_back(prevPts[i]);
            validCurrPts.push_back(currPts[i]);
            validIndices.push_back(static_cast<int>(i));
        }
    }

    // 如果初步筛选后没有点，直接返回
    if (validIndices.empty())
    {
        currFrame->mvleftpixel.clear();
        currFrame->mvFeatureIds.clear();
        currFrame->mvTrackCnt.clear();
        currFrame->iFeaturePointnums = 0;
        return;
    }
    // 反向光流校验
    if (Config::g_nFlowBack)
    {
        std::vector<cv::Point2f> backwardPts; // 从 curr 反推回 prev 的点
        std::vector<uchar> backwardStatus;
        std::vector<float> backwardErr;

        // 反向追踪：以当前帧的点为起点，追回上一帧
        cv::calcOpticalFlowPyrLK(currImg, prevImg, validCurrPts, backwardPts,
                                 backwardStatus, backwardErr, cv::Size(21, 21), 3);

        // 重新构建有效的索引（只有正向成功 && 反向成功 && 往返误差 < 1.0 像素才算内点）
        std::vector<int> validatedIndices;
        for (size_t i = 0; i < validIndices.size(); ++i)
        {
            if (backwardStatus[i]) // 反向追踪成功
            {
                // 计算原始上一帧点 与 反向推算回来的点 的欧氏距离
                double dist = cv::norm(validPrevPts[i] - backwardPts[i]);
                if (dist < 1.0) // 误差阈值 1.0 像素（可根据场景微调）
                {
                    validatedIndices.push_back(validIndices[i]);
                }
            }
        }
        validIndices = validatedIndices;
        validPrevPts.clear();
        validCurrPts.clear();
        for (int idx : validIndices)
        {
            validPrevPts.push_back(prevPts[idx]);
            validCurrPts.push_back(currPts[idx]);
        }
        // 如果双向校验后点太少，就跳过双向校验结果，继续使用原来的 validIndices（即只做正向）
    }
    // 第二步：帧间 RANSAC 剔除误匹配
    std::vector<int> inlierIdxInValid = FilterInterFrameMismatch(validPrevPts, validCurrPts);
    // 将内点索引映射回原始索引
    std::vector<int> finalIndices;
    finalIndices.reserve(inlierIdxInValid.size());
    for (int idx : inlierIdxInValid)
    {
        finalIndices.push_back(validIndices[idx]);
    }

    // 第三步：对剩余的内点进行去重（防止空间扎堆）
    std::vector<cv::KeyPoint> filteredPts;
    std::vector<int> filteredIds;
    std::vector<int> filteredCnt;
    filteredPts.reserve(finalIndices.size());
    filteredIds.reserve(finalIndices.size());
    filteredCnt.reserve(finalIndices.size());

    for (int idx : finalIndices)
    {
        cv::Point2f pt = currPts[idx];

        bool tooClose = false;
        for (const auto &exist_kp : filteredPts)
        {
            if (cv::norm(pt - exist_kp.pt) < minFeatureDist)
            {
                tooClose = true;
                break;
            }
        }
        if (tooClose)
            continue;

        filteredPts.push_back(cv::KeyPoint(pt, 1.0f));
        filteredIds.push_back(prevFrame->mvFeatureIds[idx]);
        filteredCnt.push_back(prevFrame->mvTrackCnt[idx] + 1);
    }

    // 将过滤后的数据移交给当前帧
    currFrame->mvleftpixel = std::move(filteredPts);
    currFrame->mvFeatureIds = std::move(filteredIds);
    currFrame->mvTrackCnt = std::move(filteredCnt);
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

            cv::circle(mask, kp.pt, minFeatureDist, 0, -1);
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
                                          const std::vector<int> &trackCnt,
                                          cv::Mat &outDisplay)
{
    cv::Mat colorLeft, colorRight;

    if (imgLeft.channels() == 1)
        cv::cvtColor(imgLeft, colorLeft, cv::COLOR_GRAY2BGR);
    else
        colorLeft = imgLeft.clone();

    if (imgRight.channels() == 1)
        cv::cvtColor(imgRight, colorRight, cv::COLOR_GRAY2BGR);
    else
        colorRight = imgRight.clone();

    for (size_t i = 0; i < leftKeys.size(); ++i)
    {
        int cnt = (i < trackCnt.size()) ? trackCnt[i] : 1;
        double hue = std::max(0.0, 120.0 - cnt * 5.0);
        cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(static_cast<uchar>(hue), 255, 255));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        cv::Scalar pointColor = cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[0],
                                           bgr.at<cv::Vec3b>(0, 0)[1],
                                           bgr.at<cv::Vec3b>(0, 0)[2]);
        cv::circle(colorLeft, leftKeys[i].pt, 3, pointColor, -1);

        // int cnt = (i < trackCnt.size()) ? trackCnt[i] : 1;
        // cv::Scalar pointColor;
        // if (cnt <= 1)
        // {
        //     // 新提取的特征点 -> 蓝色 (BGR = 255, 0, 0)
        //     pointColor = cv::Scalar(255, 0, 0);
        // }
        // else
        // {
        //     // 多次可观测的稳定特征点 -> 红色 (BGR = 0, 0, 255)
        //     pointColor = cv::Scalar(0, 0, 255);
        // }
        // cv::circle(colorLeft, leftKeys[i].pt, 3, pointColor, -1);
    }

    for (const auto &kp : rightKeys)
    {
        if (kp.pt.x >= 0 && kp.pt.y >= 0)
        {
            cv::circle(colorRight, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
        }
    }

    cv::vconcat(colorLeft, colorRight, outDisplay);
}

std::vector<int> FeatureDetector::FilterInterFrameMismatch(const std::vector<cv::Point2f> &prevPts,
                                                           const std::vector<cv::Point2f> &currPts)
{
    std::vector<int> inlierIndices;
    if (prevPts.size() < 8 || currPts.size() < 8)
    {
        // 点数太少，无法可靠估计基础矩阵，全部保留
        inlierIndices.resize(prevPts.size());
        std::iota(inlierIndices.begin(), inlierIndices.end(), 0);
        return inlierIndices;
    }

    std::vector<uchar> status;
    // 使用 RANSAC 求基础矩阵，重投影误差阈值 1.0 像素 (可调)
    cv::findFundamentalMat(prevPts, currPts, cv::FM_RANSAC, 1.0, 0.99, status);

    for (size_t i = 0; i < status.size(); ++i)
    {
        if (status[i])
            inlierIndices.push_back(static_cast<int>(i));
    }

    // 如果内点太少（比如 < 6），可能估计失败，退回全部保留（根据工程需求也可丢弃所有）
    if (inlierIndices.size() < 6)
    {
        inlierIndices.clear();
        inlierIndices.resize(prevPts.size());
        std::iota(inlierIndices.begin(), inlierIndices.end(), 0);
    }

    return inlierIndices;
}