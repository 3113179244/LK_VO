#include <opencv2/opencv.hpp>
#include <vector>
#include <list>

// 定义单个被追踪的特征点结构
struct TrackedFeature {
    int id;               // 全局唯一的特征点ID
    cv::Point2f ptLeft;   // 当前帧左目图像坐标
    cv::Point2f ptRight;  // 当前帧右目图像坐标 (通过双目光流或极线搜索得到)
    int trackCount;       // 该点被连续追踪的次数 (用于筛选高质量特征)
};

class FeatureTracker {
public:
    FeatureTracker();

    // 核心处理接口：传入当前帧的左右图像，返回追踪到的特征点列表
    void TrackImage(const cv::Mat& currImgLeft, const cv::Mat& currImgRight);

    // 获取当前成功追踪并完成双目匹配的特征点
    std::vector<TrackedFeature> GetCurrentTrackedFeatures() const;

private:
    // ==========================================
    // 内部核心处理函数
    // ==========================================
    
    // 1. 时序光流追踪：上一帧左目 -> 当前帧左目
    void TrackTemporal();

    // 2. 双目光流追踪：当前帧左目 -> 当前帧右目 (用于恢复深度)
    void TrackStereo();

    // 3. 异常点剔除：使用基础矩阵 (Fundamental Matrix) 和 RANSAC
    void RejectOutliers();

    // 4. 补充新特征：当追踪点过少时，在空白区域提取新角点
    void DetectNewFeatures();

    // 辅助函数：生成掩膜，防止提取的新特征点扎堆
    void SetMask();

private:
    // 图像缓存
    cv::Mat prevImgLeft;       // 上一帧左目图像
    cv::Mat currImgLeft;       // 当前帧左目图像
    cv::Mat currImgRight;      // 当前帧右目图像

    // 特征点缓存
    std::vector<cv::Point2f> prevPtsLeft; // 上一帧左目的特征点坐标
    std::vector<cv::Point2f> currPtsLeft; // 当前帧左目的特征点坐标
    std::vector<cv::Point2f> currPtsRight;// 当前帧右目的特征点坐标

    // 对应特征点的全局ID，长度与 prevPtsLeft 一致
    std::vector<int> trackIds; 
    
    // 记录每个特征点被连续追踪的帧数
    std::vector<int> trackCnt; 

    // 用于均匀化特征点分布的掩膜
    cv::Mat mask;

    // ==========================================
    // 配置参数
    // ==========================================
    int maxFeatures;           // 画面中维持的最大特征点数量 (如 150-300)
    int minFeatureDist;        // 特征点之间的最小像素距离 (如 30px)
    static int nextFeatureId;  // 全局自增的特征点生成ID
};