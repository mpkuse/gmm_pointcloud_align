#include "OpticalFlowTracker.h"


OpticalFlowTracker::OpticalFlowTracker()
{
    criteria = cv::TermCriteria(
            (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS),
            30,
            0.01);


    win_size = cv::Size(25,25);
    n_pyramid_lvl = 3;

}


OpticalFlowTracker::OpticalFlowTracker(int _window_, int _n_pyramids_ )
{
    criteria = cv::TermCriteria(
            (cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS),
            30,
            0.01);

    assert( _window_ > 5 && _window_ < 40 );
    assert( _n_pyramids_ > 0 && _n_pyramids_ < 12 );
    win_size = cv::Size(_window_,_window_);
    n_pyramid_lvl = _n_pyramids_;

}


/// Also set the features to track using goodFeaturesToTrack
#define __OpticalFlowTracker__setKeyframe__( msg ) msg;
// #define __OpticalFlowTracker__setKeyframe__( msg ) ;
void OpticalFlowTracker::setKeyframe( const cv::Mat im, int maxCorners, int min_distance_between_corners )
{
    clear_tracked_data();

    keyframe = im;

    // params for goodFeaturesToTrack
    // maxCorners = 500;
    double qualityLevel = 0.01;
    int minDistance = min_distance_between_corners; //25;
    cv::Mat mask = cv::Mat();
    int blockSize = 7;
    bool useHarrisDetector = false;
    double k = 0.04;


    cv::goodFeaturesToTrack(keyframe, kf_p,
        maxCorners, qualityLevel, minDistance,
        mask,
        blockSize,
        useHarrisDetector, k);
    MiscUtils::point2f_2_eigen( kf_p, keyframe_features_uv );
    assert( (int)kf_p.size() == keyframe_features_uv.cols() );

    m_keyframe = true;
    __OpticalFlowTracker__setKeyframe__(
    cout << "[OpticalFlowTracker::setKeyframe_1] keyframe=" << MiscUtils::cvmat_info(keyframe) << "\tdetected goodFeaturesToTrack=" << kf_p.size() << endl;
    )
}

/// Set the provided features as tracking features, features_uv will be a 2xN or 3xN matrix
void OpticalFlowTracker::setKeyframe( const cv::Mat im, const MatrixXd features_uv )
{
    clear_tracked_data();

    keyframe = im;
    assert( features_uv.rows() == 2 || features_uv.rows() == 3 );
    assert( features_uv.cols() > 0 ); //ideally must be atleast greater than say 15
    keyframe_features_uv = features_uv;
    MiscUtils::eigen_2_point2f( keyframe_features_uv, kf_p );


    m_keyframe = true;
    __OpticalFlowTracker__setKeyframe__(
    cout << "[OpticalFlowTracker::setKeyframe_2] keyframe=" << MiscUtils::cvmat_info(keyframe) << "\tinput n_features=" << features_uv.cols() << endl;
    )
}



#define __OpticalFlowTracker__trackFromKeyframe__( msg ) msg;
// #define __OpticalFlowTracker__trackFromKeyframe__( msg ) ;
int OpticalFlowTracker::trackFromKeyframe( const cv::Mat curr_im ) //< returns number of successfully tracked
{
    assert( curr_im.rows > 0 && curr_im.cols > 0 );


    status.clear();
    err.clear();
    assert( m_keyframe && "You called OpticalFlowTracker::trackFromKeyframe without setting a keyframe. You need to call setKeyframe before calling this function");

    curr_p.clear();

    __OpticalFlowTracker__trackFromKeyframe__(
    ElapsedTime t_optflow("OpticalFlowTracker"); )
    cv::calcOpticalFlowPyrLK(
        keyframe, curr_im,
        kf_p, curr_p,
        status, err,
        win_size, n_pyramid_lvl, criteria);


    __OpticalFlowTracker__trackFromKeyframe__(
    cout << "[OpticalFlowTracker::trackFromKeyframe] PyrLK tracker returned " << curr_p.size() << " pts";
    cout << " of which " << MiscUtils::total_positives(status) << " were successfully tracked";
    cout << " " << t_optflow.toc();
    cout << endl;
    )


    #if 0
    // F-test
    // TODO: compute fundamental matrix only on the successfully tracked out do MiscUtils::reduce_vector.
    //       after that set the status to 0 for thise tracked points that dont follow this F.

    vector<uchar> status_ftest;
    cv::findFundamentalMat(kf_p, curr_p, cv::FM_RANSAC, 5.0, 0.99, status_ftest);
    cout << "After F-test only " << MiscUtils::total_positives( status_ftest ) << " remain " << endl;
    #endif



    int n_successfully_tracked = MiscUtils::total_positives(status);
    return (int) n_successfully_tracked;

}





#define __OpticalFlowTracker__trackFromPrevframe__( msg ) msg;
// #define __OpticalFlowTracker__trackFromKeyframe__( msg ) ;
int OpticalFlowTracker::trackFromPrevframe( const cv::Mat curr_im ) //< returns number of successfully tracked
{
    assert( curr_im.rows > 0 && curr_im.cols > 0 );


    status.clear();
    err.clear();
    assert( m_keyframe && "You called OpticalFlowTracker::trackFromKeyframe without setting a keyframe. You need to call setKeyframe before calling this function");

    curr_p.clear();


    // if prev_p is empty set it as key frame
    if( prev_p.size() == 0 ) {
        cout << "[OpticalFlowTracker::trackFromPrevframe] WARN Since prev_p (and prev_im) is empty. This means this is the 1st input after setting keyframe, set it equal to keyframe.\n";
        prev_p = kf_p;
        prev_im = keyframe;
    }

    __OpticalFlowTracker__trackFromPrevframe__(
    ElapsedTime t_optflow("OpticalFlowTracker"); )
    cv::calcOpticalFlowPyrLK(
        prev_im, curr_im,
        prev_p, curr_p,
        status, err,
        win_size, n_pyramid_lvl, criteria);


    __OpticalFlowTracker__trackFromPrevframe__(
    cout << "[OpticalFlowTracker::trackFromPrevframe] PyrLK tracker returned " << curr_p.size() << " pts";
    cout << " of which " << MiscUtils::total_positives(status) << " were successfully tracked";
    cout << " " << t_optflow.toc();
    cout << endl;
    )


    #if 0
    // F-test
    // TODO: compute fundamental matrix only on the successfully tracked out do MiscUtils::reduce_vector.
    //       after that set the status to 0 for thise tracked points that dont follow this F.

    vector<uchar> status_ftest;
    cv::findFundamentalMat(kf_p, curr_p, cv::FM_RANSAC, 5.0, 0.99, status_ftest);
    cout << "After F-test only " << MiscUtils::total_positives( status_ftest ) << " remain " << endl;
    #endif


    prev_im = curr_im;
    prev_p = curr_p;

    int n_successfully_tracked = MiscUtils::total_positives(status);
    return (int) n_successfully_tracked;

}

const int OpticalFlowTracker::n_tracked_success( ) const
{
    assert( curr_p.size() > 0  && status.size() == curr_p.size() );

    return MiscUtils::total_positives(status);
}


const MatrixXd OpticalFlowTracker::tracked_uv( ) const
{
    int n = curr_p.size();
    assert( curr_p.size() > 0 );
    MatrixXd curr_X;
    MiscUtils::point2f_2_eigen( curr_p, curr_X );
    assert( (curr_X.rows() == 2 || curr_X.rows() == 3) && curr_X.cols() == n  );
    return curr_X;

}

const vector<cv::Point2f> OpticalFlowTracker::tracked_p() const
{
    assert( curr_p.size() > 0 );
    return curr_p;
}

const vector<uchar> OpticalFlowTracker::tracked_status_org() const
{
    assert( curr_p.size() > 0 && status.size() == curr_p.size() );
    return status;
}

const vector<bool> OpticalFlowTracker::tracked_status() const
{
    assert( curr_p.size() > 0 && status.size() == curr_p.size() );
    vector<bool> status_bool;
    for( int k=0 ; k<(int)status.size() ; k++ )
    {
        // cout << k << ":" << (int) status[k] << " : " << err[k]  << "\t" << curr_p[k];
        // cout << endl;
        if( status[k] > 0 )
            status_bool.push_back(true);
        else
            status_bool.push_back(false);

    }
    return status_bool;

}
