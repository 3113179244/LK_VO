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

// 前向声明你项目中的类
class Tracker;
class Map;
class MapPoint;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);
    ~FrameDrawer() = default;

    // 从 Tracker 更新最新处理的帧数据
    void Update(Tracker* pTracker);

    // 绘制并返回带特征点/状态信息的图像
    cv::Mat DrawFrame();

protected:
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);

    // 绘制所需缓存数据
    cv::Mat mIm;
    int N;
    std::vector<cv::KeyPoint> mvCurrentKeys;
    std::vector<bool> mvbMap, mvbVO;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    std::vector<cv::KeyPoint> mvIniKeys;
    std::vector<int> mvIniMatches;
    int mState;

    Map* mpMap;
    std::mutex mMutex;
};

#endif // FRAMEDRAWER_H