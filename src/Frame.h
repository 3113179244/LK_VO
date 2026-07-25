#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <mutex>

class Camera;    // 相机内参模型
class MapPoint;  // 3D地图点

class Frame {
public:
    // 智能指针定义，方便内存管理
    typedef std::shared_ptr<Frame> Ptr;

    // 构造函数：传入左右图像、时间戳、相机模型等
    Frame(const cv::Mat& imLeft, const cv::Mat& imRight, const double timestamp, 
          std::shared_ptr<Camera> camera, const int id);

    // --- 核心处理函数 ---
    // 提取特征点与描述子 (可以内部调用，也可以外部传入)
    void ExtractFeatures(); 
    
    // 双目立体匹配，计算视差和深度
    void ComputeStereoMatches();
    
    // 设置/获取当前帧的位姿 T_cw (世界坐标系到相机坐标系的变换矩阵)
    void SetPose(const cv::Mat& Tcw);
    cv::Mat GetPose();
    
    // 获取相机光心在世界坐标系下的位置
    cv::Mat GetCameraCenter();

    // 判断某个特征点是否在视野内
    bool isInFrustum(const MapPoint* pMP, float viewingCosLimit);

public:
    // ==========================================
    // 1. 基础信息 (Metadata)
    // ==========================================
    long unsigned int mId;       // 帧的唯一ID
    double mTimeStamp;           // 时间戳
    
    // ==========================================
    // 2. 传感器与相机参数
    // ==========================================
    std::shared_ptr<Camera> mpCamera; // 相机模型（包含焦距 fx, fy, 光心 cx, cy, 畸变参数）
    float mbf;                        // baseline * fx (基线乘以焦距，用于计算深度)
    float mThDepth;                   // 深度阈值 (区分远点和近点，远点退化为单目处理)

    // ==========================================
    // 3. 位姿 (Pose)
    // ==========================================
    // T_cw: Camera to World Transform. 
    // 常使用 SE(3) 表示，这里用 OpenCV Mat 或 Eigen::Matrix4d
    cv::Mat mTcw;      
    cv::Mat mRcw;      // 旋转矩阵 R
    cv::Mat mtcw;      // 平移向量 t
    cv::Mat mRwc;      // R^T
    cv::Mat mOw;       // 光心位置: -R^T * t

    // ==========================================
    // 4. 特征点与描述子 (核心数据)
    // ==========================================
    int N;                                 // 提取到的特征点总数量 (以左目为准)
    std::vector<cv::KeyPoint> mvKeys;      // 左目特征点 (去畸变后的坐标)
    std::vector<cv::KeyPoint> mvKeysRight; // 右目特征点 (仅在匹配时临时使用，可优化掉)
    cv::Mat mDescriptors;                  // 左目特征点的描述子

    // ==========================================
    // 5. 双目特有信息 (Stereo Specific)
    // ==========================================
    // 长度均等于 N (左目特征点数)
    std::vector<float> mvuRight; // 左目特征点在右目图像中匹配到的横坐标 u_right (如果未匹配上则为 -1)
    std::vector<float> mvDepth;  // 每个特征点对应的深度 z。 z = mbf / (u_left - u_right)

    // ==========================================
    // 6. 地图与跟踪信息 (Map Relations)
    // ==========================================
    // 长度等于 N，记录每个特征点关联的 3D 地图点 (未关联则为 nullptr)
    std::vector<std::shared_ptr<MapPoint>> mvpMapPoints;
    // 记录特征点是否为外点 (Outlier)，在优化时标记
    std::vector<bool> mvbOutlier; 
    
    // 用于特征点网格化加速匹配的栅格 (Grid)
    // std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

private:
    std::mutex mMutexPose; // 保护位姿更新的互斥锁 (多线程系统必备)
};