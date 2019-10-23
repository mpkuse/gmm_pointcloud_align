// OpenCV Optical flow from opencv optical flow tutorial.


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

const string im_path = "/app/catkin_ws/src/gmm_pointcloud_align/resources/images_for_optical_flow/";
vector<string> im_list =
{
"left_image__1562810821196652749.jpg",
"left_image__1562810821263315610.jpg",
"left_image__1562810821330322848.jpg",
"left_image__1562810821396446367.jpg",
"left_image__1562810821463562347.jpg",
"left_image__1562810821530215829.jpg",
"left_image__1562810821596585070.jpg",
"left_image__1562810821663623454.jpg",
"left_image__1562810821730188114.jpg",
"left_image__1562810821796931976.jpg"
};


#include "OpticalFlowTracker.h"
int main()
{
    cout << "OpticalFlowTracker() test\n";
    OpticalFlowTracker tracker( 21, 3 );
    // tracker.setKeyframe( )


    cout << "n images=" << im_list.size() << endl;
    for( int i=0 ; i<(int)im_list.size() ; i++ )
    {
        cout << "---\n" << im_list[i] << endl;
        cv::Mat curr_im = cv::imread( im_path + "/" + im_list[i], 0 );
        cout << "im info " << MiscUtils::cvmat_info( curr_im ) << endl;
        cout << "-\n";
        cv::imshow( "curr_im", curr_im );

        if( i%5 == 0 )
        {
            tracker.setKeyframe( curr_im, 500 );

            #if 1
            assert( tracker.isKeyFrameSet() );
            cv::Mat dst;
            MiscUtils::plot_point_sets( tracker.keyframe_image(), tracker.keyframe_uv(), dst, cv::Scalar(255,0,0),true, "n_features="+to_string(tracker.keyframe_nfeatures()) );
            cv::imshow( "goodFeaturesToTrack", dst );
            #endif
        }
        else
        {
            // int successfully_tracked = tracker.trackFromKeyframe( curr_im );
            int successfully_tracked = tracker.trackFromPrevframe( curr_im );
            // or
            // int successfully_tracked = tracker.n_tracked_success();


            #if 1
            assert( tracker.isKeyFrameSet() );
            cv::Mat dst;
            string msg ="n_features="+to_string(tracker.keyframe_nfeatures())+";successfully tracked = " + to_string(successfully_tracked) ;
            MiscUtils::plot_point_sets_masked( curr_im,
                tracker.tracked_uv(), tracker.tracked_status(),
                dst, cv::Scalar(255,0,0),true, msg );
            cv::imshow( "tracked", dst );
            #endif
        }

        cv::waitKey(0);

    }

}

#if 0
int main()
{
    cout << "helow test_lk\n";
    cout << "n images=" << im_list.size() << endl;

    cv::Mat curr_im, prev_im;
    vector<cv::Point2f> curr_p, prev_p;
    MatrixXd curr_X, prev_X;

    for( int i=0 ; i<im_list.size() ; i++ )
    {
        cout << "---\n" << im_list[i] << endl;
        curr_im = cv::imread( im_path + "/" + im_list[i], 0 );
        cout << "im info " << MiscUtils::cvmat_info( curr_im ) << endl;
        // cv::imshow( "curr_im", curr_im );

        if( i==0 )
        {
            cout << TermColor::GREEN() << "goodFeaturesToTrack" << TermColor::RESET() << endl;
            cv::goodFeaturesToTrack(curr_im, curr_p, 500, 0.3, 7, cv::Mat(), 7, false, 0.04);
            MiscUtils::point2f_2_eigen( curr_p, curr_X );
            cout << "Detected " << curr_X.cols() << " points\n";

            prev_p = curr_p;
            prev_X = curr_X;
            prev_im = curr_im;


            #if 1
            cv::Mat dst;
            MiscUtils::plot_point_sets( curr_im, curr_X, dst, cv::Scalar(255,0,0) );
            cv::imshow( "goodFeaturesToTrack", dst );
            #endif
        }
        else
        {
            // calculate optical flow
            cout << TermColor::GREEN() << "Calculate Optical Flow" << TermColor::RESET() << endl;
            vector<uchar> status;
            vector<float> err;
            cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);
            ElapsedTime t_optflow("optical flow");
            calcOpticalFlowPyrLK(prev_im, curr_im, prev_p, curr_p, status, err, cv::Size(25,25), 4, criteria);
            cout << t_optflow.toc() << endl;

            // output: curr_p, status, err;
            cout << "curr_p: " << curr_p.size() << endl;
            int nfail = 0;
            for( int k=0 ; k<status.size() ; k++ )
            {
                cout << k << ":" << (int) status[k] << " : " << err[k]  << "\t" << curr_p[k];
                cout << endl;
                if( status[k] == 0 )
                    nfail++;
            }
            cout << "nfail=" << nfail << endl;

            #if 1
            cv::Mat dst_tr;
            MiscUtils::point2f_2_eigen( curr_p, curr_X );
            MiscUtils::plot_point_sets_masked( curr_im, curr_X, status, dst_tr, cv::Scalar(0,0,255), true, "nfail="+to_string(nfail)+" of total_tracked="+to_string(status.size()) );
            cv::imshow( "dst_tr", dst_tr );
            #endif
        }

        cv::waitKey(0);
        //book keeping
        prev_im = curr_im;
        prev_p = curr_p;
        prev_X = curr_X;
    }
}
#endif
