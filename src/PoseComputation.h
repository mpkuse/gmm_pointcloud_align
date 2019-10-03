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
    // BMVC1995 (closed form 3d align SVD, 4 algos compared, especially see section 2.1)`
    // Solves: minimize_{a_T_b}   || aX - a_T_b*bX ||_2
    // NOTE: This does not need an initial guess. so b_T_a is a just output.
    //      aX : 3xN or 4xN 3d points in co-ordinates system of `a`
    //      bX : 3xN or 4xN 3d points in co-ordinates system of `b`
    //      b_T_a [output] : resulting pose between the co-ordinates. Pose of a as observed from b.
    static bool closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b );


    static bool align3D3DWithRefinement( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b );
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
    EuclideanDistanceResidueSwitchingConstraint( const Vector3d& Xi, const Vector3d& Xid, const double lambda=3.0 )
    {
        this->Xi = Xi;
        this->Xid = Xid;
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
        residual[3] = T(3.) * (T(1.0) - s[0]);

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
};
