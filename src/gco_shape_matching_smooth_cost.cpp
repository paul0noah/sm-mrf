#include "gco_shape_matching.hpp"

namespace smgco {

GCoptimization::EnergyTermType smoothFnGCOSMTrianglewise(GCoptimization::SiteID s1,
                                                         GCoptimization::SiteID s2,
                                                         GCoptimization::LabelID l1,
                                                         GCoptimization::LabelID l2,
                                                         void* extraDataVoid) {
    GCOTrianglewiseExtra* extraData = static_cast<GCOTrianglewiseExtra*>(extraDataVoid);
    if (USE_CACHING) {
        const auto& it = extraData->cache.find(std::make_tuple(std::min(l1, l2), std::max(l1, l2)));
        if (it != extraData->cache.end()) {
            return it->second;
        }
    }

    const TriangleWiseOpts opts = extraData->opts;
    const COST_MODE costMode = extraData->opts.costMode;

    float diff = 0.0f;

    if (costMode == SINGLE_LABLE_SPACE_L2) {
        diff = extraData->p2pDeformation(l1, l2);
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_L2) {
        const std::tuple<int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenSites);
        const int idxX2 = std::get<1>(commonVerticesBetweenSites);
        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);

        const Eigen::Vector3f translation1 = extraData->translationsXtoY(rowIndex1, colIndex1);
        const Eigen::Vector3f translation2 = extraData->translationsXtoY(rowIndex2, colIndex2);

        Eigen::MatrixXf defTri1 = (Eigen::MatrixXf(2, 3) << extraData->VX.row(idxX1), extraData->VX.row(idxX2)).finished();
        Eigen::MatrixXf defTri2 = defTri1.rowwise() + translation2.transpose();
        defTri1 = defTri1.rowwise() + translation1.transpose();

        const float diff11_21 = (defTri1.row(0) - defTri2.row(0)).norm();
        const float diff11_22 = (defTri1.row(0) - defTri2.row(1)).norm();
        const float diff12_21 = (defTri1.row(1) - defTri2.row(0)).norm();
        const float diff12_22 = (defTri1.row(1) - defTri2.row(1)).norm();

        diff = std::max({diff11_21, diff11_22, diff12_21, diff12_22});

    }
    else if (costMode == MULTIPLE_LABLE_SPACE_SO3 || costMode == MULTIPLE_LABLE_SPACE_GEODIST_SO3) {
        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);
        const Eigen::Quaterniond rotation1 = extraData->quaternionsXtoY(rowIndex1, colIndex1);
        const Eigen::Quaterniond rotation2 = extraData->quaternionsXtoY(rowIndex2, colIndex2);
        double innerProductRot = fabs(rotation1.w() * rotation2.w() + rotation1.vec().dot(rotation2.vec()));
        innerProductRot = std::max(std::min(1.0, innerProductRot), -1.0); // clip into value range [-1, ..., 1]

        diff += opts.lambdaSo3 * 2 * acos(innerProductRot);
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_SE3) {
        const int rowIndex1 = extraData->lableToIndex(l1, 0);
        const int rowIndex2 = extraData->lableToIndex(l2, 0);
        const int colIndex1 = extraData->lableToIndex(l1, 1);
        const int colIndex2 = extraData->lableToIndex(l2, 1);
        const Eigen::Quaterniond rotation1 = extraData->quaternionsXtoY(rowIndex1, colIndex1);
        const Eigen::Matrix3f rot1 = rotation1.toRotationMatrix().cast<float>();
        const Eigen::Quaterniond rotation2 = extraData->quaternionsXtoY(rowIndex2, colIndex2);
        const Eigen::Matrix3f rot2 = rotation2.toRotationMatrix().cast<float>();

        const std::tuple<int, int> commonVerticesBetweenSites = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenSites);
        const int idxX2 = std::get<1>(commonVerticesBetweenSites);

        const Eigen::Vector3f translation1 = extraData->translationsXtoY(rowIndex1, colIndex1);
        const Eigen::Vector3f translation2 = extraData->translationsXtoY(rowIndex2, colIndex2);

        double innerProductRot = fabs(rotation1.w() * rotation2.w() + rotation1.vec().dot(rotation2.vec()));
        innerProductRot = std::max(std::min(1.0, innerProductRot), -1.0); // clip into value range [-1, ..., 1]

        const Eigen::Vector3f vert1 = extraData->VX.row(idxX1);
        const Eigen::Vector3f vert2 = extraData->VX.row(idxX2);

        const Eigen::Vector3f t1Minust2 = translation1 - translation2;

        const Eigen::Vector3f defVert1_1 = rot1 * vert1;
        const Eigen::Vector3f defVert2_1 = rot1 * vert2;

        const Eigen::Vector3f defVert1_2 = rot2 * vert2;
        const Eigen::Vector3f defVert2_2 = rot2 * vert2;

        const float se3_11_12 = (defVert1_1 - defVert1_2 + t1Minust2).norm();
        const float se3_11_22 = (defVert1_1 - defVert2_2 + t1Minust2).norm();
        const float se3_21_12 = (defVert2_1 - defVert1_2 + t1Minust2).norm();
        const float se3_21_22 = (defVert2_1 - defVert2_2 + t1Minust2).norm();

        const float maxse3dist = std::max({se3_11_12, se3_11_22, se3_21_12, se3_21_22});

        const float so3dist = 2 * acos(innerProductRot);

        diff = opts.lambdaSe3 * maxse3dist + opts.lambdaSo3 * so3dist;
    }
    else if (costMode == MULTIPLE_LABLE_SPACE_GEODIST ||
             costMode == MULTIPLE_LABLE_SPACE_GEODIST_MAX ||
             costMode == MULTIPLE_LABLE_SPACE_L2DIST ||
             costMode == MULTIPLE_LABLE_SPACE_L2DIST_MAX ||
             costMode == MULTIPLE_LABLE_SPACE_GEODIST_SO3) {
        const std::tuple<int, int> commonVerticesBetweenTriangles = extraData->commonVXofFX(s1, s2);
        const int idxX1 = std::get<0>(commonVerticesBetweenTriangles);
        const int idxX2 = std::get<1>(commonVerticesBetweenTriangles);
        const int realLableIndex1 = extraData->lableToIndex(l1, 1);
        const int realLableIndex2 = extraData->lableToIndex(l2, 1);
        int colIdx11 = -1, colIdx12 = -1, colIdx21 = -1, colIdx22 = -1;
        for (int j = 0; j < 3; j++) {
            if (extraData->FX(s1, j) == idxX1) colIdx11 = j;
            if (extraData->FX(s1, j) == idxX2) colIdx12 = j;
            if (extraData->FX(s2, j) == idxX1) colIdx21 = j;
            if (extraData->FX(s2, j) == idxX2) colIdx22 = j;
        }
        const int targetVertex1_1 = extraData->LableFY(realLableIndex1, colIdx11);
        const int targetVertex1_2 = extraData->LableFY(realLableIndex1, colIdx12);
        const int targetVertex2_1 = extraData->LableFY(realLableIndex2, colIdx21);
        const int targetVertex2_2 = extraData->LableFY(realLableIndex2, colIdx22);

        if (costMode == MULTIPLE_LABLE_SPACE_L2DIST_MAX || costMode == MULTIPLE_LABLE_SPACE_GEODIST_MAX) {
            diff = std::max(extraData->geoDistY(targetVertex1_1, targetVertex2_1), extraData->geoDistY(targetVertex1_2, targetVertex2_2));
        }
        else {
            diff = extraData->geoDistY(targetVertex1_1, targetVertex2_1) + extraData->geoDistY(targetVertex1_2, targetVertex2_2);
        }

        if (opts.sameLabelCost > 0) {
            // this ensures that we have a metric, in theory we would need an if statement but the case that l1 == l2 cannot happen for us
            diff += opts.sameLabelCost;
        }
    }

    diff = opts.smoothScaleBeforeRobust * diff;

    if (opts.robustCost == 1) {
        diff = std::log(1 + diff);
    }
    else if (opts.robustCost >= 5) {
        // hinge like loss but concave
        if (diff > extraData->robustMinThres) {
            const float newslope = extraData->newSlope;
            const float oneminusnewslope = 1.0f - newslope;
            diff = newslope * diff + oneminusnewslope * extraData->robustMinThres;
        }
    }
    else if (opts.robustCost > 1) {
        diff = std::min(extraData->robustMinThres, diff);
    }

    diff = opts.smoothWeight * diff;

    const long long output = (long long) (SCALING_FACTOR * diff );
    if (USE_CACHING) {
        extraData->cache.insert({std::make_tuple(std::min(l1, l2), std::max(l1, l2)), output});
    }

    return output;
}

} // namespace smgco
