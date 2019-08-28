#pragma once



//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <math.h>
#include <vector>

#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory> //needed for std::shared_ptr
using namespace std;

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"

#include "utils/MiscUtils.h"

#include "utils/TermColor.h"




/*
class SurfelElement {
public:
    Vector3d wX;
    Vector3d normal;


};
*/

class SurfelXMap {
public:
    SurfelXMap( camodocal::CameraPtr _camera );
    bool clear_data();

    bool fuse_with( int i, Matrix4d __wTc, MatrixXd __sp_cX, MatrixXd __sp_uv, cv::Mat& image_i, cv::Mat& depth_i ) ;

private:
    vector<int> camIdx;
    vector<Matrix4d> w_T_c; //TODO: it is unnecessary to store current superpixel related info.
    vector<MatrixXd> sp_cX; //TODO: it is unnecessary to store current superpixel related info.
    vector<MatrixXd> sp_uv; //TODO: it is unnecessary to store current superpixel related info.


public:
    int surfelSize() const;
    Vector4d surfelWorldPosition(int i) const; //returns i'th 3d pt
    MatrixXd surfelWorldPosition() const; //returns all 3d points in db. returns 4xN matrix
    // Vector3d surfelSurfaceNormal(int i); // TODO

private:
    mutable std::mutex * surfel_mutex;

    // Surfels
    MatrixXd S__wX;
    MatrixXd S__normal;
    int S__size = 0;
    vector<int> n_fused; ///< number of times this surfel was used, len of this shoudl be S__size
    vector<int> n_unstable; ///< number of times this surfel was found to be unstable, when projecting on the current view, len of this shoudl be S__size

private:

    // projection related
    camodocal::CameraPtr m_camera;
    void perspectiveProject3DPoints( const MatrixXd& c_V, MatrixXd& c_v ); //TODO: remove


    // Find nearest neighbours (radius search) of vector b, 3x1 or 4x1 in the database A 3xN or 4xN
    bool find_nn_of_b_in_A( const MatrixXd& A, const VectorXd& b, const double radius, vector<int>& to_retidx ) const;

};
