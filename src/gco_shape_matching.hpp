#ifndef GCO_SHAPE_MATCHING
#define GCO_SHAPE_MATCHING
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <gco/GCoptimization.h>
#include "helper/shape.hpp"
#include <unordered_map>
#define USE_CACHING false

#define SCALING_FACTOR 10000.0f
#define GETTIME(x) std::chrono::steady_clock::time_point x = std::chrono::steady_clock::now()
#define DURATION_MS(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count()
#define DURATION_S(x, y) std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count() / 1000
#define PRINT_SMGCO(x) std::cout << prefix << x << std::endl;

int omp_thread_count();
int getThreadId();

typedef Eigen::MatrixX<std::tuple<int, int>> TupleMatrixInt;

namespace smgco {
enum COST_MODE {
    SINGLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_L2,
    MULTIPLE_LABLE_SPACE_SO3,
    MULTIPLE_LABLE_SPACE_SE3,
    MULTIPLE_LABLE_SPACE_GEODIST,
    MULTIPLE_LABLE_SPACE_GEODIST_MAX,
    MULTIPLE_LABLE_SPACE_L2DIST,
    MULTIPLE_LABLE_SPACE_L2DIST_MAX,
    MULTIPLE_LABLE_SPACE_GEODIST_SO3,
};

struct TriangleWiseOpts {
    COST_MODE costMode = COST_MODE::MULTIPLE_LABLE_SPACE_GEODIST_MAX;
    float smoothScaleBeforeRobust = 0.1;
    int robustCost = 8; // 0: no, 1: log(1 + smooth), 2: min(geodistMean, smooth), 3: min(0.25 * geodistMax, smooth), 4: min(0.1 * geodistMax, smooth), 5-7: edge-length based, 8: auto
    int setInitialLables = 4;
    float lambdaSe3 = 1.0;
    float lambdaSo3 = 1.0;
    float unaryWeight = 1.0;
    float smoothWeight = 1000.0;
    int lableSpaceCycleSize = 4;
    float lableSpaceAngleThreshold = M_PI / 3;
    bool lableSpaceDegenerate = true;
    float membraneFactor = 0.0f;
    float bendingFactor = 0.0f;
    float wksFactor = 0.0f;
    float featureFactor = 1.0f;
    bool glueSolution = true;
    int labelOrder = 0; // 0 no order, 1 random, 2 degenerate last, 3 mincost, 4 alternating min cost
    float sameLabelCost = 0.0;
    int algorithm = 5; // 0 = alpha beta swap, 1 = alpha expansion, 4 = custom alpha expansion, 5 = custom alpha expansion with adaptive cycles, 6 = custom alpha expansion with site reordering
    int sinkhornIters = 10;
    float sinkhornEntropy = -1.0f;
    bool sinkhornEnergyMod = false;
    bool bicolouring = false;
};

typedef struct GCOTrianglewiseExtra {
    // not all of the below matrices are needed for all cost modes
    float lambda;
    const TriangleWiseOpts& opts;
    unsigned long numLables;
    float robustMinThres;
    float newSlope = 0.001f;
    Eigen::MatrixXf p2pDeformation;
    Eigen::MatrixXf VX;
    Eigen::MatrixXi FX;
    Eigen::MatrixXi LableFY;
    TupleMatrixInt commonVXofFX;
    Eigen::MatrixX<Eigen::Quaterniond> quaternionsXtoY;
    Eigen::MatrixX<Eigen::Vector3f> translationsXtoY;
    Eigen::MatrixXi lableToIndex;
    Eigen::MatrixXf geoDistY;
    std::unordered_map<std::tuple<GCoptimization::LabelID, GCoptimization::LabelID>, int> cache;
    std::unordered_map<std::tuple<int, int>, std::tuple<std::tuple<int, int>, std::tuple<int, int>>> sharedVertIds;
    GCOTrianglewiseExtra(TriangleWiseOpts& opts) : opts(opts) {}
} GCOTrianglewiseExtra;



class GCOSM {
    private:
    std::string prefix;
    int numIters;
    float dataWeight;
    Eigen::MatrixXd VX;
    const Eigen::MatrixXi FX;
    Eigen::MatrixXd VY;
    const Eigen::MatrixXi FY;
    const Eigen::MatrixXd perVertexFeatureDifference; // should be |VX| x |VY|
    void triangleWiseInit(TriangleWiseOpts& opts,
                          GCOTrianglewiseExtra& extraSmooth,
                          GCoptimizationGeneralGraph *gc,
                          const Eigen::MatrixXi& AdjFX,
                          Eigen::MatrixXi& minLables,
                          const Eigen::MatrixXf& softP,
                          const unsigned long numDegenerate);
    void triangleWiseAlgorithm(TriangleWiseOpts& opts,
                               GCOTrianglewiseExtra& extraSmooth,
                               GCoptimizationGeneralGraph *gc,
                               const Eigen::MatrixXi& AdjFX,
                               const Eigen::MatrixXi& siteLabelCostInt,
                               const Eigen::MatrixXi& ColourFX,
                               const int numBlue);
    unsigned long long computeEnergy(GCoptimizationGeneralGraph* gc,
                                     const Eigen::MatrixXi& AdjFX,
                                     const Eigen::MatrixXi& siteLabelCostInt,
                                     void* extraData);

    public:
    GCOSM(const Eigen::MatrixXd VX,
          const Eigen::MatrixXi FX,
          const Eigen::MatrixXd VY,
          const Eigen::MatrixXi FY,
          const Eigen::MatrixXd perVertexFeatureDifference);
    void printName();
    void updatePrefix(const std::string& newprefix);

    Eigen::MatrixXi pointWise(const bool smoothGeodesic=false);
    std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> triangleWise(TriangleWiseOpts opts);
    std::tuple<float, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi, Eigen::MatrixXi> triangleWise();

    void setDataWeight(const float newDataWeight);
    void setMaxIter(const int newMaxIter);

    void writeToFile(const std::string& filename);
};


Eigen::MatrixXi glueResult(const Eigen::MatrixXi& result,
                           const Eigen::MatrixXd& VX,
                           const Eigen::MatrixXi& FX,
                           const Eigen::MatrixXd& VY,
                           const Eigen::MatrixXi& FY,
                           const Eigen::MatrixXf& geoDistY=Eigen::MatrixXf(0, 0));

void precomputeSmoothCost(const Eigen::MatrixXd& VX,
                          const Eigen::MatrixXi& FX,
                          const Eigen::MatrixXd& VY,
                          const Eigen::MatrixXi& FY,
                          const Eigen::MatrixXi& lableSpace,
                          GCOTrianglewiseExtra& extraData);

Eigen::MatrixXi buildLableSpace(const Eigen::MatrixXd& VY,
                                const Eigen::MatrixXi& FY,
                                unsigned long& numDegenerate,
                                TriangleWiseOpts& opts);

GCoptimization::EnergyTermType smoothFnGCOSMTrianglewise(GCoptimization::SiteID s1,
                                                         GCoptimization::SiteID s2,
                                                         GCoptimization::LabelID l1,
                                                         GCoptimization::LabelID l2,
                                                         void* extraDataVoid);

Eigen::MatrixXi getAllNewNeighLabels(const int f,
                                     const int l,
                                     const Eigen::MatrixXi& AdjFX,
                                     const std::vector<std::vector<int>>& vertexInLables,
                                     const GCOTrianglewiseExtra& extraSmooth);

} // namespace smgco


#endif // GCO_SHAPE_MATCHING
