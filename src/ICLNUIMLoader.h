#pragma once
// this class will help you load ICL_NUIM dataset


#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

// opencv2

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"
#include "utils/RawFileIO.h"
#include "utils/PoseManipUtils.h"

class ICLNUIMLoader
{
public:
    ICLNUIMLoader( const string DB_BASE, const string DB_NAME, const int DB_IDX );
    void print_info( ) const;

    bool retrive_im( int i, cv::Mat& im );
    bool retrive_im_depth( int i, cv::Mat& im, cv::Mat& depth );
    bool retrive_im_depth( int i, cv::Mat& im, cv::Mat& depth, cv::Mat& depth_falsecolor );
    bool retrive_pose( int i, Matrix4d& wTc );
    ros::Time idx_to_stamp( int i );

    int len() const { return min(IM_LIST.size(), POSE_LIST.size()); }

    const string DB_BASE;
    const string DB_NAME;
    const int DB_IDX;

private:
    string img_fld, traj_file;
    ros::Time begin;

    vector<string> IM_LIST;
    vector<string> DEPTH_LIST;
    vector< Matrix4d > POSE_LIST;

    bool imread_depth( string fname, cv::Mat& depth );



public:
    bool load_gt_pose();  // this will load the file: livingRoom0n.gt.sim etc, only available for livingRoom sequence

private:
    vector<Matrix4d> GT_POSE_LIST;
    bool m_gt_poses_available = false;


public:
    void set_camera_intrinsics( float _fx, float _fy, float _cx, float _cy )
    {
        fx = _fx; fy = _fy; cx = _cx; cy = _cy;
        m_cameraIntrinsics = true;
    }
private:
    float fx, fy, cx, cy;
    bool m_cameraIntrinsics =false;
};
