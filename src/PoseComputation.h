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

// Ceres
#include <ceres/ceres.h>
using namespace ceres;

class PoseComputation
{
public:
    // Given two point sets computes the relative transformation using a
    // closed form SVD based method. implementation is based on
    // `A  Comparison  of Four  Algorithms  forEstimating  3-D Rigid  Transformations,
    // BMVC1995 (closed form 3d align SVD, 4 algos compared, especially see section 2.1, Arun et al. TPAMI1987 )`
    // Solves: minimize_{a_T_b}   || aX - a_T_b*bX ||_2
    // NOTE: This does not need an initial guess. so b_T_a is a just output.
    //      aX : 3xN or 4xN 3d points in co-ordinates system of `a`
    //      bX : 3xN or 4xN 3d points in co-ordinates system of `b`
    //      b_T_a [output] : resulting pose between the co-ordinates. Pose of a as observed from b.
    static bool closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b );

    // Given 2 point sets and an initial guess of the alignment iteratively refine the estimate
    // The problem is setup as non-linear least squares with robust norm and switching constraints.
    // Solves: minimize_{a_T_b, s1,s2,...sn} \sum_i s_i * || aX_i - a_T_b*bX_i ||_2 +  \lambda*(1-s_i)
    //      s1,s2,... are scalars between [0,1], \lambda is a fixed scalar
    //      aX : 3xN or 4xN 3d points in co-ordinates system of `a`
    //      bX : 3xN or 4xN 3d points in co-ordinates system of `b`
    //      b_T_a [input/output] : initial guess. resulting pose between the co-ordinates. Pose of a as observed from b.
    static bool refine( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b );


    //----
    // Given 2 point sets (correspondences) and depth values separately along with the weights
    // of each correspondences, computes the relative pose without an initial guess (ie. in closed form)
    // The method is an adaptation of Arun et al TPAMI1986.
    //  Solves: minimize_{a_T_b}  ( aX - a_T_b*bX )^T  *  W  * ( aX - a_T_b*bX )^T
    //              where aX := aX_sans_depth.topRows(3) .* d_a
    //                    bX := bX_sans_depth.topRows(3) .* d_b
    static bool closedFormSVD( const MatrixXd& aX_sans_depth, const VectorXd& d_a,
                               const MatrixXd& bX_sans_depth, const VectorXd& d_b,
                               const VectorXd& sf,
                               Matrix4d& a_T_b
                           );

// starts with initial guesses (of pose, switch-weights, depths) refine these. Can set any of these as
// constants depending on the flag
//      aX_sans_depth : 3xN Matrix of normalized image co-ordinates. ie. it the 3d point once you multiply depth (z) to it.
//                          wT0.inverse() * wTa * uv_a_normalized
//      d_a, [input/output] : The depth values. These are depths in co-ordinate system of camera
//      bX_sans_depth : 3xN same as aX_sans_depth
//                            w'_T_0 * w'Tb * uv_b_normalized
//      d_b, [input/output] : The depth values. These are depths in co-ordinate system of camera
//      sf [input/output] : switch constraints
//      a_T_b[input/output] : the transform between the 2 co-ordinate systems.
//      refine_pose [flags] : true to optimize the pose a_T_b
//      refine_depth
//      refine_switch_weights
    static bool refine( const MatrixXd& aX_sans_depth, VectorXd& d_a,
                       const MatrixXd& bX_sans_depth, VectorXd& d_b,
                       VectorXd& sf,
                       Matrix4d& a_T_b,
                       bool refine_pose, bool refine_depth, bool refine_switch_weights
                          );



};


//---------------------------------------------------//
//------------ Ceres Cost Functions -----------------//
//---------------------------------------------------//


class EuclideanDistanceResidue {
public:
    EuclideanDistanceResidue( const Vector3d& Xi, const Vector3d& Xid )
    {
        this->Xi = Xi;
        this->Xid = Xid;
    }

    template <typename T>
    bool operator()( const T* const q, const T* const t , T* residual ) const {
        // Optimization variables
        Quaternion<T> eigen_q( q[0], q[1], q[2], q[3] );
        Eigen::Matrix<T,3,1> eigen_t;
        eigen_t << t[0], t[1], t[2];


        // Known Constant
        Eigen::Matrix<T,3,1> eigen_Xi, eigen_Xid;
        eigen_Xi << T(Xi(0)), T(Xi(1)), T(Xi(2));
        eigen_Xid << T(Xid(0)), T(Xid(1)), T(Xid(2));



        // Error term
        Eigen::Matrix<T,3,1> e;
        e = eigen_Xi - (  eigen_q.toRotationMatrix() * eigen_Xid + eigen_t );

        residual[0] = e(0);
        residual[1] = e(1);
        residual[2] = e(2);

        return true;
    }



    static ceres::CostFunction* Create(const Vector3d& _Xi, const Vector3d& Xid)
    {
        return ( new ceres::AutoDiffCostFunction<EuclideanDistanceResidue,3,4,3>
        (
        new EuclideanDistanceResidue(_Xi,Xid)
        )
        );
    }

    private:
    Vector3d Xi, Xid;
};





class EuclideanDistanceResidueSwitchingConstraint  {
public:
    // lambda is the switch-penalty.
    EuclideanDistanceResidueSwitchingConstraint( const Vector3d& Xi, const Vector3d& Xid, const double lambda_=3.0 )
    {
        this->Xi = Xi;
        this->Xid = Xid;
        this->lambda = lambda_;
    }

    template <typename T>
    bool operator()( const T* const q, const T* const t , const T* const s, T* residual ) const {
        // Optimization variables
        Quaternion<T> eigen_q( q[0], q[1], q[2], q[3] );
        Eigen::Matrix<T,3,1> eigen_t;
        eigen_t << t[0], t[1], t[2];


        // Known Constant
        Eigen::Matrix<T,3,1> eigen_Xi, eigen_Xid;
        eigen_Xi << T(Xi(0)), T(Xi(1)), T(Xi(2));
        eigen_Xid << T(Xid(0)), T(Xid(1)), T(Xid(2));



        // Error term

        Eigen::Matrix<T,3,1> e;
        e = eigen_Xi - (  eigen_q.toRotationMatrix() * eigen_Xid + eigen_t );

        residual[0] = s[0] * e(0);
        residual[1] = s[0] * e(1);
        residual[2] = s[0] * e(2);
        residual[3] = T(lambda) * (T(1.0) - s[0]);

        return true;
    }

    static ceres::CostFunction* Create(const Vector3d& _Xi, const Vector3d& Xid)
    {
        return ( new ceres::AutoDiffCostFunction<EuclideanDistanceResidueSwitchingConstraint,4,   4,3,1>
          (
            new EuclideanDistanceResidueSwitchingConstraint(_Xi,Xid)
          )
        );
    }

private:
    Vector3d Xi, Xid;
    double lambda;
};




class EuclideanDistanceResidueSwitchingConstraintAndDepthRefinement  {
public:
    // lambda is the switch-penalty.
    EuclideanDistanceResidueSwitchingConstraintAndDepthRefinement( const Vector3d& Xi, const Vector3d& Xid, const double lambda_=3.0 )
    {
        this->Xi = Xi;
        this->Xid = Xid;
        this->lambda = lambda_;
    }

    template <typename T>
    bool operator()( const T* const q, const T* const t , const T* const s, const T* d_a, const T* d_b, T* residual ) const {

        /*
        if( s[0] < T(0.1) ) {
            residual[0] = T(0);
            residual[1] = T(0);
            residual[2] = T(0);
            residual[3] = T(0);
            return true;
        }*/

        // Optimization variables
        Quaternion<T> eigen_q( q[0], q[1], q[2], q[3] );
        Eigen::Matrix<T,3,1> eigen_t;
        eigen_t << t[0], t[1], t[2];


        // Known Constant
        Eigen::Matrix<T,3,1> eigen_Xi, eigen_Xid;
        eigen_Xi << *d_a * T(Xi(0)), *d_a * T(Xi(1)), *d_a *  T(Xi(2));
        eigen_Xid << *d_b * T(Xid(0)), *d_b * T(Xid(1)), *d_b * T(Xid(2));



        // Error term

        Eigen::Matrix<T,3,1> e;
        e = eigen_Xi - (  eigen_q.toRotationMatrix() * eigen_Xid + eigen_t );

        residual[0] = s[0] * e(0);
        residual[1] = s[0] * e(1);
        residual[2] = s[0] * e(2);
        residual[3] = T(lambda) * (T(1.0) - s[0]);

        return true;
    }

    static ceres::CostFunction* Create(const Vector3d& _Xi, const Vector3d& Xid)
    {
        return ( new ceres::AutoDiffCostFunction<EuclideanDistanceResidueSwitchingConstraintAndDepthRefinement,4,   4,3,1, 1,1>
          (
            new EuclideanDistanceResidueSwitchingConstraintAndDepthRefinement(_Xi,Xid)
          )
        );
    }

private:
    Vector3d Xi, Xid; // here these are 3vector representing normalized image co-ordinates. You need to multiply these with depths to get the 3d points
    double lambda;
};
