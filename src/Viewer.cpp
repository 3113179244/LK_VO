#include "Viewer.h"
#include "Map.h"       
#include "Frame.h"     
#include "MapPoint.h"
#include <iostream>

Viewer::Viewer(std::shared_ptr<Map> pMap) 
    : mpMap(pMap), mbFinishRequested(false), mbFinished(false)
{
    mKeyFrameSize = 0.05f;
    mKeyFrameLineWidth = 1.0f;
    mPointSize = 2.0f;
    mCameraSize = 0.08f;
    mCameraLineWidth = 3.0f;
    // 初始化当前位姿为单位矩阵
    mCurrentTcw = Eigen::Matrix4d::Identity();
}

void Viewer::Run()
{
    // 1. 创建Pangolin窗口
    pangolin::CreateWindowAndBind("LK_VO: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 2. 设置相机参数（视角）
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.7, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    // 3. 创建交互视图
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (1)
    {
        // 清屏
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 白色背景

        d_cam.Activate(s_cam);

        // 绘制地图元素
        DrawMapPoints();
        DrawKeyFrames();
        DrawCurrentCamera();

        pangolin::FinishFrame();

        // 检查终止请求
        std::unique_lock<std::mutex> lock(mMutexFinish);
        if (mbFinishRequested)
        {
            mbFinished = true;
            break;
        }
        lock.unlock();

        // 小延时，避免CPU占用过高
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void Viewer::DrawMapPoints()
{
    // 从Map中获取所有地图点
    std::vector<std::shared_ptr<MapPoint>> vpMPs = mpMap->GetAllMapPoints();
    if (vpMPs.empty()) return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0); // 黑色

    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        if (vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetMapPoints();
        // 注意：MapPoint存储的是CV_32F格式
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }
    glEnd();
}

void Viewer::DrawKeyFrames()
{
    // 获取所有关键帧（Frame）
    std::vector<std::shared_ptr<Frame>> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty()) return;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        // 获取位姿（Eigen::Matrix4f）
        Eigen::Matrix4f Tcw = pKF->GetPose();
        // 用蓝色绘制关键帧相机（RGB: 0, 0.6, 1）
        DrawCamera(Tcw, 0.0f, 0.6f, 1.0f, mCameraSize);
    }
}

void Viewer::DrawCurrentCamera()
{
    std::unique_lock<std::mutex> lock(mMutexCurrentCam);
    // 用绿色绘制当前帧相机（RGB: 0, 1, 0）
    Eigen::Matrix4f Tcw = mCurrentTcw.cast<float>();
    DrawCamera(Tcw, 0.0f, 1.0f, 0.0f, mCameraSize * 1.2f);
}

void Viewer::DrawCamera(const Eigen::Matrix4f& Tcw, float r, float g, float b, float scale)
{
    // 计算相机在世界坐标系下的位姿（Twc = Tcw.inverse()）
    Eigen::Matrix4f Twc = Tcw.inverse();
    Eigen::Vector3f c = Twc.block<3,1>(0,3);   // 光心坐标
    Eigen::Matrix3f R = Twc.block<3,3>(0,0);   // 旋转矩阵

    glPushMatrix();
    glTranslatef(c.x(), c.y(), c.z());

    // 将旋转矩阵转换为OpenGL列优先矩阵
    GLfloat mat[16];
    mat[0] = R(0,0); mat[1] = R(1,0); mat[2] = R(2,0); mat[3] = 0;
    mat[4] = R(0,1); mat[5] = R(1,1); mat[6] = R(2,1); mat[7] = 0;
    mat[8] = R(0,2); mat[9] = R(1,2); mat[10] = R(2,2); mat[11] = 0;
    mat[12] = 0;      mat[13] = 0;      mat[14] = 0;      mat[15] = 1;
    glMultMatrixf(mat);

    // 绘制相机模型（金字塔形）
    float sz = scale * 0.5f;
    float depth = scale * 0.8f;
    // 四个角点（相机坐标系中，z正向为前方，即视线方向）
    Eigen::Vector3f p[4] = {
        Eigen::Vector3f(-sz, -sz, depth),
        Eigen::Vector3f( sz, -sz, depth),
        Eigen::Vector3f( sz,  sz, depth),
        Eigen::Vector3f(-sz,  sz, depth)
    };
    Eigen::Vector3f origin(0,0,0);

    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3f(r, g, b);
    // 从光心到四个角的连线
    for (int i=0; i<4; i++) {
        glVertex3f(origin(0), origin(1), origin(2));
        glVertex3f(p[i](0), p[i](1), p[i](2));
    }
    // 矩形边框
    for (int i=0; i<4; i++) {
        glVertex3f(p[i](0), p[i](1), p[i](2));
        glVertex3f(p[(i+1)%4](0), p[(i+1)%4](1), p[(i+1)%4](2));
    }
    glEnd();

    // 绘制坐标轴（红X，绿Y，蓝Z），长度稍长
    glLineWidth(1);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0,0,0); glVertex3f(scale, 0, 0);
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0,0,0); glVertex3f(0, scale, 0);
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0,0,0); glVertex3f(0, 0, scale);
    glEnd();

    glPopMatrix();
}

void Viewer::SetCurrentCameraPose(const Eigen::Matrix4d& Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexCurrentCam);
    mCurrentTcw = Tcw;
}

void Viewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}