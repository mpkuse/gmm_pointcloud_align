#pragma once

// Optical flow
// Will have 2 modes: a) Tracking from keyframe
// tracking from previous frame


#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

// opencv2
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>


// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"
#include "utils/MiscUtils.h"


class OpticalFlowTracker
{
public:
    OpticalFlowTracker();
    OpticalFlowTracker(int _window_, int _n_pyramids_ );

    //-----------
    void setKeyframe( const cv::Mat im,  int maxCorners=200, int min_distance_between_corners=25 ); //< Also set the features to track using goodFeaturesToTrack
    void setKeyframe( const cv::Mat im, const MatrixXd features_uv ); //< Set the provided features as tracking features, features_uv will be a 2xN or 3xN matrix

    bool isKeyFrameSet() { return m_keyframe; }
    const cv::Mat keyframe_image() const { assert(m_keyframe); return keyframe; }
    const MatrixXd keyframe_uv() const { assert( m_keyframe); return keyframe_features_uv; }
    const vector<cv::Point2f> keyframe_p() const { assert( m_keyframe); return kf_p; }
    const int keyframe_nfeatures() { assert(m_keyframe); return (int)kf_p.size(); }


    //---------------
    int trackFromKeyframe( const cv::Mat curr_im ); //< returns number of successfully tracked
    int trackFromPrevframe( const cv::Mat curr_im ); //< track from previous frame, as you provide it images, it will keep prev_im in memory

    const MatrixXd tracked_uv( ) const;
    const vector<cv::Point2f> tracked_p() const;
    const vector<bool> tracked_status() const;
    const vector<uchar> tracked_status_org() const;
    const int n_tracked_success( ) const;



    //---------------
    // trackFromPreviousFrame();

private:
    // Tracker Params
    cv::TermCriteria criteria;
    cv::Size win_size;
    int n_pyramid_lvl;

    // Keyframe Data
    cv::Mat keyframe;
    MatrixXd keyframe_features_uv;
    vector<cv::Point2f> kf_p;
    bool m_keyframe=false;

    // tracked data
    vector<cv::Point2f> curr_p;
    vector<uchar> status;
    vector<float> err;

    cv::Mat prev_im;
    vector<cv::Point2f> prev_p;

    void clear_tracked_data()
    {
        curr_p.clear();
        status.clear();
        err.clear();

        prev_im = cv::Mat();
        prev_p.clear(); 
    }



};
