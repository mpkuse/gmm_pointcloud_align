#pragma once
// This class will contain all my pose computation methods, global as well as refinement based
// or any other based on normals

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>
#include <memory> //for std::shared_ptr
using namespace std;

#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"
#include "utils/PoseManipUtils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


class PoseComputation
{
public:
    // Given two point sets computes the relative transformation using a
    // closed form SVD based method. implementation is based on
    // `A  Comparison  of Four  Algorithms  forEstimating  3-D Rigid  Transformations,
    // BMVC1995 (closed form 3d align SVD, 4 algos compared, especially see section 2.1)`
    // Solves: minimize_{a_T_b}   || aX - a_T_b*bX ||_2
    //      aX : 3xN or 4xN 3d points in co-ordinates system of `a`
    //      bX : 3xN or 4xN 3d points in co-ordinates system of `b`
    //      b_T_a [output] : resulting pose between the co-ordinates. Pose of a as observed from b.
    static bool closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b );

};
