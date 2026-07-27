#ifndef INITIALIZER_H
#define INITIALIZER_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

// 前向声明
class Frame;
class Camera;
class Map;

class Initializer {
public:
    Initializer();
    ~Initializer() = default;

    bool TryInitialize(std::shared_ptr<Frame> pFrame);

private:
    std::shared_ptr<Map> mpMap;
    std::shared_ptr<Camera> mpCamera;
};

#endif // INITIALIZER_H