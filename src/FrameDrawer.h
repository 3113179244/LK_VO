#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include <vector>
#include <mutex>
#include <string>
#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>

class Tracker;
class Map;
class MapPoint;

/**
 * @brief FrameDrawer 类：负责将当前帧的图像、提取的特征点、匹配关系以及状态文字等信息绘制并输出
 */
class FrameDrawer
{
public:
    // 构造函数：绑定地图指针
    FrameDrawer(Map* pMap);
    ~FrameDrawer() = default;

    /**
     * @brief 从 Tracker 更新最新处理的帧数据（线程安全）
     * @param pTracker 跟踪器指针
     */
    void Update(Tracker* pTracker);

    /**
     * @brief 绘制并返回带特征点/状态信息的图像（给可视化UI调用）
     * @return cv::Mat 绘制好的彩色图像
     */
    cv::Mat DrawFrame();

protected:
    /**
     * @brief 在图像下方拼接黑边并绘制系统当前状态文字
     * @param im 输入图像
     * @param nState 当前系统的跟踪状态
     * @param imText 输出拼接文字后的图像
     */
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
    
    cv::Mat mIm;                            // 当前帧的图像缓存
    int N;                                  // 当前帧关键点数量
    std::vector<cv::KeyPoint> mvCurrentKeys;// 当前帧的所有关键点
    std::vector<bool> mvbMap;               // 关键点标志位：是否匹配到了建图地图点 (MapPoint)
    std::vector<bool> mvbVO;                // 关键点标志位：是否匹配到了临时里程计点 (Visual Odometry Point)
    bool mbOnlyTracking;                    // 是否开启纯定位/纯跟踪模式
    int mnTracked;                          // 成功跟踪到的地图点数量
    int mnTrackedVO;                        // 成功跟踪到的 VO 点数量
    
    // 单目初始化阶段的数据缓存
    std::vector<cv::KeyPoint> mvIniKeys;    // 初始化第一帧的关键点
    std::vector<int> mvIniMatches;          // 第一帧与当前帧关键点的匹配关系
    int mState;                             // 系统的当前跟踪状态（如 OK, LOST, NOT_INITIALIZED 等）

    Map* mpMap;                             // 全局地图指针
    std::mutex mMutex;                      // 互斥锁，保证 Update 与 DrawFrame 的线程安全
};

#endif // FRAMEDRAWER_H