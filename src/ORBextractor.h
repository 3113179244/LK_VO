#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <opencv2/opencv.hpp>

class ORBextractor
{
public:
    ORBextractor();

    int GetLevels() { return nlevels; }
    float GetScaleFactor() { return scaleFactor; }
    std::vector<float> GetScaleFactors() { return mvScaleFactor; }
    std::vector<float> GetScaleSigmaSquares() { return mvLevelSigma2; }
    std::vector<float> GetInverseScaleSigmaSquares() { return mvInvLevelSigma2; }

protected:
    int nlevels;
    float scaleFactor;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;
};

#endif // ORBEXTRACTOR_H