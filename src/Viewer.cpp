#include "Viewer.h"
#include "Map.h"
#include "Frame.h"
#include "MapPoint.h"
#include <unistd.h> // for usleep
#include <iostream>

Viewer::Viewer(std::shared_ptr<Map> pMap) 
    : mpMap(pMap), mbFinishRequested(false), mbFinished(false),
      mbStopRequested(false), mbStopped(false),
      mbFollowCamera(true), mbShowPoints(true), mbShowKeyFrames(true), mbShowGraph(true),
      mbResetRequested(false)
{
    mKeyFrameSize = 0.05f;
    mKeyFrameLineWidth = 1.0f;
    mGraphLineWidth = 1.0f;
    mPointSize = 2.0f;
    mCameraSize = 0.08f;
    mCameraLineWidth = 3.0f;
    mCurrentTcw = Eigen::Matrix4d::Identity();
}

// 【新增辅助函数】获取 OpenGL 列优先格式的当前相机 Pose (Twc)
pangolin::OpenGlMatrix Viewer::GetCurrentOpenGLCameraMatrix()
{
    Eigen::Matrix4d Twc;
    {
        std::unique_lock<std::mutex> lock(mMutexCurrentCam);
        Twc = mCurrentTcw.inverse();
    }

    pangolin::OpenGlMatrix M;
    // Pangolin 为列优先 (Column-Major)，必须按正确的行列顺序填入 M.m
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            M.m[j * 4 + i] = static_cast<float>(Twc(i, j));
        }
    }
    return M;
}

void Viewer::Run()
{
    // 1. 创建 Pangolin 窗口
    pangolin::CreateWindowAndBind("LK_VO: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 2. 创建菜单面板
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
    pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
    pangolin::Var<bool> menuReset("menu.Reset", false, false);

    // 3. 相机渲染初始视角
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0));

    // 4. 创建显示视图
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    bool bFollow = true; // 内部跟踪跟随状态标识

    while (1)
    {
        // 清屏（白色背景）
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        // 同步 UI 菜单状态
        mbFollowCamera = menuFollowCamera;
        mbShowPoints = menuShowPoints;
        mbShowKeyFrames = menuShowKeyFrames;
        mbShowGraph = menuShowGraph;

        // 获取用于跟随的 OpenGL 矩阵
        pangolin::OpenGlMatrix Twc_gl = GetCurrentOpenGLCameraMatrix();

        // 核心改动：跟随镜头（Follow Camera）逻辑
        if (mbFollowCamera && bFollow)
        {
            s_cam.Follow(Twc_gl);
        }
        else if (mbFollowCamera && !bFollow)
        {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0));
            s_cam.Follow(Twc_gl);
            bFollow = true;
        }
        else if (!mbFollowCamera && bFollow)
        {
            bFollow = false;
        }

        // 激活视图
        d_cam.Activate(s_cam);

        // 绘制地图元素
        if (mbShowPoints)
            DrawMapPoints();
        if (mbShowKeyFrames || mbShowGraph)
        {
            DrawKeyFrames(); // 绘制关键帧相机
            if (mbShowGraph)
                DrawGraph(); // 绘制轨迹连线
        }
        DrawCurrentCamera(); // 绘制当前实时相机

        // 提交 3D 渲染
        pangolin::FinishFrame();

        // 处理 Reset 重置请求
        if (menuReset)
        {
            menuFollowCamera = true;
            menuShowPoints = true;
            menuShowKeyFrames = true;
            menuShowGraph = true;

            mbFollowCamera = true;
            mbShowPoints = true;
            mbShowKeyFrames = true;
            mbShowGraph = true;
            bFollow = true;

            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0));
            
            {
                std::unique_lock<std::mutex> lock(mMutexReset);
                mbResetRequested = true;
            }
            menuReset = false;
        }

        // 检查暂停请求
        if (Stop())
        {
            while (isStopped())
            {
                usleep(3000);
            }
        }

        // 检查结束请求
        if (CheckFinish())
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    SetFinish();
}

void Viewer::DrawMapPoints()
{
    std::vector<std::shared_ptr<MapPoint>> vpMPs = mpMap->GetAllMapPoints();
    if (vpMPs.empty()) return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0f, 0.0f, 0.0f); // 黑色点云

    for (size_t i = 0; i < vpMPs.size(); i++)
    {
        if (vpMPs[i]->isBad())
            continue;
        cv::Mat pos = vpMPs[i]->GetMapPoints();
        glVertex3f(pos.at<float>(0), pos.at<float>(1), pos.at<float>(2));
    }
    glEnd();
}

void Viewer::DrawKeyFrames()
{
    std::vector<std::shared_ptr<Frame>> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.empty()) return;

    for (size_t i = 0; i < vpKFs.size(); i++)
    {
        std::shared_ptr<Frame> pKF = vpKFs[i];
        Eigen::Matrix4f Tcw = pKF->GetPose();
        DrawCamera(Tcw, 0.0f, 0.6f, 1.0f, mCameraSize);
    }
}

void Viewer::DrawGraph()
{
    std::vector<std::shared_ptr<Frame>> vpKFs = mpMap->GetAllKeyFrames();
    if (vpKFs.size() < 2) return;

    glLineWidth(mGraphLineWidth);
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    for (size_t i = 0; i < vpKFs.size() - 1; i++)
    {
        Eigen::Vector3f Ow1 = vpKFs[i]->GetCameraCenter();
        Eigen::Vector3f Ow2 = vpKFs[i+1]->GetCameraCenter();
        glVertex3f(Ow1.x(), Ow1.y(), Ow1.z());
        glVertex3f(Ow2.x(), Ow2.y(), Ow2.z());
    }
    glEnd();
}

void Viewer::DrawCurrentCamera()
{
    std::unique_lock<std::mutex> lock(mMutexCurrentCam);
    DrawCamera(mCurrentTcw.cast<float>(), 0.0f, 1.0f, 0.0f, mCameraSize * 1.2f);
}

void Viewer::DrawCamera(const Eigen::Matrix4f& Tcw, float r, float g, float b, float scale)
{
    Eigen::Matrix4f Twc = Tcw.inverse();
    Eigen::Vector3f c = Twc.block<3,1>(0,3);
    Eigen::Matrix3f R = Twc.block<3,3>(0,0);

    glPushMatrix();
    glTranslatef(c.x(), c.y(), c.z());

    GLfloat mat[16];
    mat[0] = R(0,0); mat[1] = R(1,0); mat[2] = R(2,0); mat[3] = 0;
    mat[4] = R(0,1); mat[5] = R(1,1); mat[6] = R(2,1); mat[7] = 0;
    mat[8] = R(0,2); mat[9] = R(1,2); mat[10] = R(2,2); mat[11] = 0;
    mat[12] = 0;      mat[13] = 0;      mat[14] = 0;      mat[15] = 1;
    glMultMatrixf(mat);

    float sz = scale * 0.5f;
    float depth = scale * 0.8f;
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
    for (int i=0; i<4; i++) {
        glVertex3f(origin(0), origin(1), origin(2));
        glVertex3f(p[i](0), p[i](1), p[i](2));
    }
    for (int i=0; i<4; i++) {
        glVertex3f(p[i](0), p[i](1), p[i](2));
        glVertex3f(p[(i+1)%4](0), p[(i+1)%4](1), p[(i+1)%4](2));
    }
    glEnd();

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

void Viewer::SetCurrentFrameImage(const cv::Mat& img)
{
    std::unique_lock<std::mutex> lock(mMutexImg);
    img.copyTo(mCurrentFrameImg);
}

void Viewer::RequestReset()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    mbResetRequested = true;
}

bool Viewer::CheckReset()
{
    std::unique_lock<std::mutex> lock(mMutexReset);
    bool ret = mbResetRequested;
    mbResetRequested = false;
    return ret;
}

void Viewer::RequestFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

void Viewer::RequestStop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    if (!mbStopped)
        mbStopRequested = true;
}

void Viewer::Release()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    mbStopped = false;
}

bool Viewer::isFinished()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinished;
}

bool Viewer::isStopped()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
{
    std::unique_lock<std::mutex> lock(mMutexStop);
    std::unique_lock<std::mutex> lock2(mMutexFinish);
    if (mbFinishRequested)
        return false;
    if (mbStopRequested)
    {
        mbStopped = true;
        mbStopRequested = false;
        return true;
    }
    return false;
}

bool Viewer::CheckFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    std::unique_lock<std::mutex> lock(mMutexFinish);
    mbFinished = true;
}