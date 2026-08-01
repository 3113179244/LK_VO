#include "ORBextractor.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>

const int PATCH_SIZE = 31;         // 计算描述子所用的图像块尺寸 31x31
const int HALF_PATCH_SIZE = 15;    // 图像块的半长，中心点到边界的距离
const int EDGE_THRESHOLD = 19;     // 边缘阈值，靠近图像边缘的区域不提取特征，留出足够空间计算描述子

static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

/**
 * @brief 将一个节点（图像块）均分为四个子节点
 * 
 * 计算出原节点的中心点，以此将矩形分为左上、右上、左下、右下四个小矩形。
 * 然后遍历原节点内的所有特征点，根据坐标分配到对应的子节点中。
 */
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    // 计算当前节点长宽的一半
    const float halfX = (UR.x - UL.x) / 2.0f;
    const float halfY = (BR.y - UL.y) / 2.0f;

    // 分配子节点 1：左上
    n1.UL = UL; n1.UR = cv::Point2f(UL.x + halfX, UL.y);
    n1.BL = cv::Point2f(UL.x, UL.y + halfY); n1.BR = cv::Point2f(UL.x + halfX, UL.y + halfY);

    // 分配子节点 2：右上
    n2.UL = n1.UR; n2.UR = UR;
    n2.BL = n1.BR; n2.BR = cv::Point2f(UR.x, UL.y + halfY);

    // 分配子节点 3：左下
    n3.UL = n1.BL; n3.UR = n1.BR;
    n3.BL = BL; n3.BR = cv::Point2f(UL.x + halfX, BL.y);

    // 分配子节点 4：右下
    n4.UL = n1.BR; n4.UR = n2.BR;
    n4.BL = n3.BR; n4.BR = BR;

    // 将父节点中的特征点分配到 4 个子节点中
    for(size_t i = 0; i < vKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x < n1.UR.x) { 
            // 在左半边
            if(kp.pt.y < n1.BL.y) n1.vKeys.push_back(kp); // 左上
            else n3.vKeys.push_back(kp);                  // 左下
        } else {
            // 在右半边
            if(kp.pt.y < n1.BL.y) n2.vKeys.push_back(kp); // 右上
            else n4.vKeys.push_back(kp);                  // 右下
        }
    }

    // 如果子节点包含的特征点数量 <= 1，则该节点不可再分，标记为 bNoMore
    if(n1.vKeys.size() <= 1) n1.bNoMore = true;
    if(n2.vKeys.size() <= 1) n2.bNoMore = true;
    if(n3.vKeys.size() <= 1) n3.bNoMore = true;
    if(n4.vKeys.size() <= 1) n4.bNoMore = true;
}

/**
 * @brief ORB 提取器构造函数
 * 
 * 预计算尺度因子、尺度倒数以及每一层预期需要提取的特征点数量等常量。
 */
ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels), iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    // 初始化每层的尺度因子和尺度方差
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;
    for (int i = 1; i < nlevels; i++) {
        mvScaleFactor[i] = mvScaleFactor[i - 1] * scaleFactor;
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    // 初始化尺度因子的倒数，主要用于后续坐标转换等操作，提前计算除法变乘法可加速
    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for (int i = 0; i < nlevels; i++) {
        mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
    }

    // 分配每层预期提取的特征点数：原则上按照每层的图像面积比例分配
    mvImagePyramid.resize(nlevels);
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    // 根据等比数列求和公式，计算第 0 层应该分配多少个特征点
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - pow(factor, nlevels));

    int sumFeatures = 0;
    for (int level = 0; level < nlevels - 1; level++) {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor; // 后一层的目标特征数等于前一层乘以尺度倒数（面积比例）
    }
    // 最后一层分配剩余的特征点，确保总数严格等于 nfeatures
    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

    // 初始化 BRIEF 描述子的 256 组采样点模式（这里用 extern 数组 bit_pattern_31_ 提供预定义好的坐标对）
    const int npoints = 512;
    const cv::Point* ptMat = (const cv::Point*)bit_pattern_31_;
    pattern.resize(npoints);
    std::copy(ptMat, ptMat + npoints, pattern.begin());

    // 预计算一个圆形的掩码，用于灰度质心法的计算。只在半径为 HALF_PATCH_SIZE 的圆形区域内计算。
    // umax[v] 表示在 y=v 这一行上，圆的右边界的 x 坐标
    umax.resize(HALF_PATCH_SIZE + 1);
    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE * HALF_PATCH_SIZE;
    
    // 利用圆的方程 u^2 + v^2 = r^2 计算 1/8 圆周的边界
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));
    
    // 根据对称性计算余下部分的边界，确保圆形平滑
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v) {
        while (umax[v0] == umax[v0 + 1]) ++v0;
        umax[v] = v0;
        ++v0;
    }
}

/**
 * @brief 灰度质心法计算特征点方向
 * 
 * 质心法认为局部图像块的光度中心偏离了几何中心，连接几何中心与光度中心的向量定义为该特征点的方向。
 */
static float IC_Angle(const cv::Mat& image, cv::Point2f pt, const std::vector<int> & umax)
{
    int m_01 = 0, m_10 = 0; // 一阶矩
    const uchar* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x)); // 图像块中心指针

    // 处理 v=0 （即中心所在的中间行），计算其水平方向的矩
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    int step = (int)image.step1(); // 获取图像每行的字节数
    // 利用对称性上下两行一起计算 v = 1 到 15
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        int v_sum = 0;
        int d = umax[v]; // 获取该行 u (即 x 方向) 的边界
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step];  // 下半圆像素
            int val_minus = center[u - v * step]; // 上半圆像素
            v_sum += (val_plus - val_minus);      // 垂直分量差
            m_10 += u * (val_plus + val_minus);   // 水平分量和
        }
        m_01 += v * v_sum; // 累计计算垂直方向的矩
    }
    // 根据矩利用 atan2 求解角度返回（结果为角度值，非弧度）
    return cv::fastAtan2((float)m_01, (float)m_10);
}

/**
 * @brief 计算所有关键点的 ORB (Steered BRIEF) 描述子
 */
void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const std::vector<cv::Point>& pattern)
{
    // 初始化描述子矩阵，行数为关键点数量，列数为 32 (CV_8UC1，每行 256 位)
    descriptors = cv::Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const cv::KeyPoint& kpt = keypoints[i];
        // 将角度转为弧度
        float angle = kpt.angle * (float)CV_PI / 180.f;
        float a = (float)cos(angle), b = (float)sin(angle); // 计算旋转矩阵因子

        const uchar* center = &image.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        const int step = (int)image.step1();

        // 宏定义：获取基于特征点主方向旋转后的采样点对应位置的像素值
        #define GET_VALUE(idx) \
            center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
                   cvRound(pattern[idx].x * a - pattern[idx].y * b)]

        uchar* desc = descriptors.ptr<uchar>((int)i);
        // BRIEF 描述子长 256 bits (32 个字节)。循环 32 次，每次计算 1 个字节 (8 bits)。
        for (int j = 0; j < 32; ++j)
        {
            uchar val = 0;
            for (int k = 0; k < 8; ++k)
            {
                int idx = (j << 3) + k;
                // 获取一对预定义的采样点经过旋转后的像素值进行对比
                uchar t0 = GET_VALUE(idx * 2);
                uchar t1 = GET_VALUE(idx * 2 + 1);
                // 像素比较，如果 t0 < t1 则当前位置 1，否则置 0
                val |= (t0 < t1) << k;
            }
            desc[j] = val; // 将这 8 次比较的结果保存成 1 个字节
        }
        #undef GET_VALUE
    }
}

/**
 * @brief 构建图像金字塔
 * 根据尺度因子逐层降采样，并在图像周围补充边界(padding)以满足描述子计算不越界。
 */
void ORBextractor::ComputePyramid(const cv::Mat& image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level]; // 当前层对应缩放的尺寸比例
        cv::Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        // wholeSize 是真正分配的图像矩阵大小，包含了外层为了提取边缘特征点添加的 padding (扩边)
        cv::Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);
        cv::Mat temp(wholeSize, image.type());

        // mvImagePyramid 保存的是剔除扩边后的图像 ROI (核心图像区域)
        mvImagePyramid[level] = temp(cv::Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // 第 0 层：直接将原图拷贝过去，并做边界反射填充 padding
        if (level == 0)
            cv::copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, cv::BORDER_REFLECT_101);
        else // 后续层：基于上一层的 ROI 区域进行线性下采样 (resize)
            cv::resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
    }
}

/**
 * @brief 主操作入口：提取 ORB 关键点及计算描述子
 */
void ORBextractor::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<cv::KeyPoint>& _keypoints, cv::OutputArray _descriptors)
{
    if (_image.empty()) return;
    cv::Mat image = _image.getMat();

    // 1. 构建金字塔
    ComputePyramid(image);

    std::vector<std::vector<cv::KeyPoint>> allKeypoints(nlevels);
    // 2. 利用 FAST 算法 + 四叉树均匀分布策略，提取每一层的特征点，并计算方向角
    ComputeKeyPointsOctree(allKeypoints);

    // 统计总特征点数以分配描述子矩阵空间
    cv::Mat descriptors;
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    if (nkeypoints == 0)
        _descriptors.release();
    else
    {
        _descriptors.create(nkeypoints, 32, CV_8UC1);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    // 3. 逐层计算描述子，并将每一层的特征点尺度还原至第 0 层的坐标系
    for (int level = 0; level < nlevels; ++level)
    {
        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        int nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0) continue;

        // 对金字塔当前层图像应用高斯模糊，降低噪点对描述子的影响
        cv::Mat workingMat = mvImagePyramid[level].clone();
        cv::GaussianBlur(workingMat, workingMat, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        // 获取对应的描述子矩阵行指针，并计算 Steered BRIEF 描述子
        cv::Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc, pattern);

        offset += nkeypointsLevel;
        float scale = mvScaleFactor[level];
        
        // 恢复关键点的各种属性，尤其是把坐标通过乘以当前层尺度因子映射回第 0 层坐标
        for (std::vector<cv::KeyPoint>::iterator keypoint = keypoints.begin(), keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->pt *= scale;                // 映射回原图的坐标
            keypoint->octave = level;             // 记录所在的金字塔层数
            keypoint->size = PATCH_SIZE * scale;  // 计算在原图中对应的特征尺寸
        }
        // 追加至输出的关键点列表
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

/**
 * @brief 在金字塔每一层，通过分格的方式提取大量候选特征点，为下一步的四叉树均匀化准备数据
 */
void ORBextractor::ComputeKeyPointsOctree(std::vector<std::vector<cv::KeyPoint>>& allKeypoints)
{
    for (int level = 0; level < nlevels; ++level)
    {
        // 计算图像在当前层的有效提取边界范围
        const int minBorderX = EDGE_THRESHOLD - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        const int maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        std::vector<cv::KeyPoint> vToDistributeKeys; // 存放该层提取的所有候选 FAST 角点
        vToDistributeKeys.reserve(nfeatures * 10);

        const float width = (maxBorderX - minBorderX);
        const float height = (maxBorderY - minBorderY);

        // 将图像划分为 30x30 像素的网格进行初步提取
        const int nCols = width / 30;
        const int nRows = height / 30;
        const int wCell = ceil(width / nCols);
        const int hCell = ceil(height / nRows);

        // 遍历所有网格
        for (int i = 0; i < nRows; i++)
        {
            const float iniY = minBorderY + i * hCell;
            float maxY = iniY + hCell + 6; // 稍微扩大 6 像素做网格重叠检测，防止遗漏边界点

            if (iniY >= maxBorderY - 3) continue;
            if (maxY > maxBorderY) maxY = maxBorderY;

            for (int j = 0; j < nCols; j++)
            {
                const float iniX = minBorderX + j * wCell;
                float maxX = iniX + wCell + 6;
                if (iniX >= maxBorderX - 3) continue;
                if (maxX > maxBorderX) maxX = maxBorderX;

                std::vector<cv::KeyPoint> vKeysCell;
                // 首先用较严格的阈值 (iniThFAST) 提取角点
                cv::FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                         vKeysCell, iniThFAST, true);

                // 若严格阈值提取不到点，则采用更宽松的退化阈值 (minThFAST) 重新提取
                if (vKeysCell.empty())
                {
                    cv::FAST(mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX),
                             vKeysCell, minThFAST, true);
                }

                // 将提取到网格局部的特征点坐标转换到当前层全图坐标并收集
                if (!vKeysCell.empty())
                {
                    for (cv::KeyPoint& kpt : vKeysCell)
                    {
                        kpt.pt.x += j * wCell;
                        kpt.pt.y += i * hCell;
                        vToDistributeKeys.push_back(kpt);
                    }
                }
            }
        }

        std::vector<cv::KeyPoint>& keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        // 将收集的大量候选点交由四叉树进行筛选和均匀化分布，控制输出的数量为预期的 N 个
        keypoints = DistributeOctree(vToDistributeKeys, minBorderX, maxBorderX,
                                     minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

        // 恢复图像真实坐标，并计算每个点的角度
        const int scaledPatchSize = PATCH_SIZE;

        for (cv::KeyPoint& kpt : keypoints)
        {
            kpt.pt.x += minBorderX;
            kpt.pt.y += minBorderY;
            // 采用灰度质心法计算当前关键点方向
            kpt.angle = IC_Angle(mvImagePyramid[level], kpt.pt, umax);
        }
    }
}

/**
 * @brief 使用四叉树对候选特征点进行空间均匀化剔除
 * 
 * 核心思想：不断把图像所在区域平均分成四个象限，把特征点分配到对应的子区域内。
 * 当某区域内没有特征点，则直接删除该节点。
 * 当节点数达到期望特征数，或者无法再分时，从每个节点内挑选 Harris 响应值最大的那个特征点保留。
 */
std::vector<cv::KeyPoint> ORBextractor::DistributeOctree(const std::vector<cv::KeyPoint>& vToDistributeKeys,
                                                         const int &minX, const int &maxX, const int &minY, const int &maxY,
                                                         const int &N, const int &level)
{
    // 根据图像的长宽比，计算出最开始需要多少个根节点（通常是一行并排的几个正方形）
    const int nIni = round((float)(maxX - minX) / (maxY - minY));
    const float hX = (float)(maxX - minX) / nIni;

    std::list<ExtractorNode> lNodes;        // 使用链表存储节点，方便中间频繁执行插入、删除
    std::vector<ExtractorNode*> vpIniNodes; // 保存初始的根节点指针
    vpIniNodes.resize(nIni);

    // 生成初始节点
    for (int i = 0; i < nIni; i++)
    {
        ExtractorNode ni;
        ni.UL = cv::Point2f(hX * i, 0);
        ni.UR = cv::Point2f(hX * (i + 1), 0);
        ni.BL = cv::Point2f(ni.UL.x, maxY - minY);
        ni.BR = cv::Point2f(ni.UR.x, ni.BL.y);
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    // 将所有初始候选特征点分配到这些根节点中
    for (size_t i = 0; i < vToDistributeKeys.size(); i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        int nMinUn = std::min(int(kp.pt.x / hX), nIni - 1);
        vpIniNodes[nMinUn]->vKeys.push_back(kp);
    }

    // 初步清理：将只有 1 个点的节点打上 bNoMore 标记，直接剔除没有任何点的空节点
    auto lit = lNodes.begin();
    while (lit != lNodes.end())
    {
        if (lit->vKeys.size() == 1) {
            lit->bNoMore = true;
            lit++;
        } else if (lit->vKeys.empty()) {
            lit = lNodes.erase(lit);
        } else {
            lit++;
        }
    }

    bool bFinish = false;
    // 临时记录那些可以继续分割的节点
    std::vector<std::pair<int, ExtractorNode*>> vSizeAndPointer;
    vSizeAndPointer.reserve(lNodes.size() * 4);

    // 四叉树核心循环展开过程
    while (!bFinish)
    {
        int prevSize = lNodes.size(); // 记录本次扩展前的节点总数
        lit = lNodes.begin();
        int nToExpand = 0;
        vSizeAndPointer.clear();

        // 遍历当前的叶子节点
        while (lit != lNodes.end())
        {
            if (lit->bNoMore) {
                // 不可再分的节点直接跳过
                lit++;
            } else {
                // 如果节点还可以再分，则将其拆分成 n1, n2, n3, n4 四个子节点
                ExtractorNode n1, n2, n3, n4;
                lit->DivideNode(n1, n2, n3, n4);

                // 如果子节点里有提取到特征点，就将其加入列表最前端
                if (n1.vKeys.size() > 0) {
                    lNodes.push_front(n1);
                    if (n1.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointer.push_back(std::make_pair(n1.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                // 同理处理另外三个子节点
                if (n2.vKeys.size() > 0) {
                    lNodes.push_front(n2);
                    if (n2.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointer.push_back(std::make_pair(n2.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n3.vKeys.size() > 0) {
                    lNodes.push_front(n3);
                    if (n3.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointer.push_back(std::make_pair(n3.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if (n4.vKeys.size() > 0) {
                    lNodes.push_front(n4);
                    if (n4.vKeys.size() > 1) {
                        nToExpand++;
                        vSizeAndPointer.push_back(std::make_pair(n4.vKeys.size(), &lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                // 原父节点已被拆分替代，从链表中删除它
                lit = lNodes.erase(lit);
            }
        }

        // 终止条件：当叶子节点的数量已经大于预期需要的特征点数 N，或者所有节点都无法再继续拆分时
        if ((int)lNodes.size() >= N || (int)lNodes.size() == prevSize) {
            bFinish = true;
        }
    }

    // 最终阶段：从每个四叉树叶子节点里，选出响应值最大的 1 个特征点作为最终保留结果
    std::vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(N);

    for (auto node : lNodes)
    {
        auto &vNodeKeys = node.vKeys;
        cv::KeyPoint *pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;
        // 寻找节点内 FAST 角点响应值(Response)最大的点
        for (size_t k = 1; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}