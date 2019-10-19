#pragma once

// methods for triangulation go here.
// code borrowed from : http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>
#include <memory> //for std::shared_ptr
using namespace std;

// My utilities (mpkuse)
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"
#include "utils/PoseManipUtils.h"

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


class Triangulation
{
public:
    // Triangulate : Iteratve Refinement
    // Params:
    //      u [input]: homogenous image point (u,v,1), input in normalized image co-ordinates
    //      P [input]: Pose of the 1st camera frame
    //      u1[input]: homogenous image point (u',v',1), input in normalized image co-ordinates
    //      P1[input]: Pose of the 2nd camera frame
    // Returns:
    //      The depth at this point
    static double IterativeLinearLSTriangulation( const Vector3d& u,  const Matrix4d& P,
                                        const Vector3d& u1,    const Matrix4d& P1   );

    // Triangulate : Basic Least squares
    // Params:
    //      u [input]: homogenous image point (u,v,1), input in normalized image co-ordinates
    //      P [input]: Pose of the 1st camera frame
    //      u1[input]: homogenous image point (u',v',1), input in normalized image co-ordinates
    //      P1[input]: Pose of the 2nd camera frame
    //      Returns the triangulated 3d point in homogenous co-ordinates
    static void LinearLSTriangulation( const Vector3d& u,  const Matrix4d& P,
                                        const Vector3d& u1,    const Matrix4d& P1, Vector4d& result_X   );


    // Triangulation of a tracked point using multiple tracks from multiple images.
    // This also needs the pose of the cameras where it was tracked.
    //      ...P--P--P--base--P--P....
    // pose of base is assumed to be Identity. The returned 3d point will be in cord-ref of this base.
    // Params:
    //      base_u [input]: the tracked image point expressed in normalized image co-ordinates
    //      p_T_base [input]: pose of the base as observed from the camera where this point was tracked
    //      result_X [output]: the resulting 3d point in cord-ref of base
    //      status [input, optional]: in which 'p's to ignore. If this is not given, we will use all.
    static void MultiViewLinearLSTriangulation( const Vector3d& base_u,
        const vector<Matrix4d>& p_T_base, Vector4d& result_X,  const vector<bool>status=vector<bool>() );
};
