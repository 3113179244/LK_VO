#ifndef MOTIONONLYBA_H
#define MOTIONONLYBA_H

class Frame;

class MotionOnlyBA
{
public:
    /**
     * @brief 仅优化当前帧的位姿 (Pose-Only BA)
     * @param pFrame 当前帧指针
     * @return 优化后的内点 (Inlier) 数量
     */
    static int Optimize(Frame *pFrame);
};

#endif // MOTIONONLYBA_H