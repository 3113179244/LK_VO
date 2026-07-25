#include <opencv2/opencv.hpp>
#include <map>
#include <memory>
#include <mutex>

class Frame; // 前向声明

class MapPoint {
public:
    // 智能指针定义
    typedef std::shared_ptr<MapPoint> Ptr;

    // 构造函数：传入全局唯一ID和初始计算的3D坐标
    // 这里的 id 应该直接使用 FeatureTracker 中生成的全局特征点 ID
    MapPoint(const long unsigned int id, const cv::Mat& position);

    // ==========================================
    // 1. 3D 位置的读写 (会被前端 Tracking 和后端 BA 优化并发访问)
    // ==========================================
    void SetWorldPos(const cv::Mat& Pos);
    cv::Mat GetWorldPos();

    // ==========================================
    // 2. 观测图维护 (共视关系)
    // ==========================================
    // 记录哪一帧 (pFrame) 的第几个特征点 (idx) 观测到了这个3D点
    void AddObservation(std::shared_ptr<Frame> pFrame, size_t idx);
    void RemoveObservation(std::shared_ptr<Frame> pFrame);
    
    // 获取所有的观测记录
    std::map<std::shared_ptr<Frame>, size_t> GetObservations();
    
    // 获取当前观测到该点的总帧数 (用于判断点的质量)
    int GetObservedCount();

    // ==========================================
    // 3. 状态管理
    // ==========================================
    // 标记该点为坏点 (Outlier)，例如重投影误差过大时
    void SetBadFlag();
    bool isBad();

public:
    const long unsigned int mId; // 全局唯一ID，与光流特征点ID强绑定

private:
    cv::Mat mWorldPos;           // 世界坐标系下的 3D 坐标 (3x1矩阵)
    
    // 核心数据结构：观测图。
    // Key是观测到该点的Frame指针，Value是该点在Frame特征列表中的索引
    std::map<std::shared_ptr<Frame>, size_t> mObservations; 
    
    int mnObs;                   // 观测计数器
    bool mbBad;                  // 坏点标记

    // 多线程互斥锁：SLAM系统极其容易在这里发生 Data Race
    std::mutex mMutexPos;        // 保护 3D 位置
    std::mutex mMutexFeatures;   // 保护观测图的增删改
};