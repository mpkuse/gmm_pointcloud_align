// monocular depth


#include <iostream>
#include <vector>
#include <fstream>
#include <map>
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


#include "XLoader.h"

#include "utils/PointFeatureMatching.h"
#include "OpticalFlowTracker.h"
#include "Triangulation.h"

void print_usage()
{
    cout << "k: set this as keyframe\t";
    cout << "a: track\t";
    cout << "ESC: quit\t";
    cout << endl;
}

int main()
{
    //
    // Load Camera (camodocal)
    //
    XLoader xloader;
    xloader.load_left_camera();
    xloader.load_right_camera();
    xloader.load_stereo_extrinsics();
    xloader.make_stereogeometry();


    //
    // Load JSON
    //
    json STATE = xloader.load_json();

    OpticalFlowTracker tracker( 21, 3 );


    Matrix4d w_T_kf;
    cv::Mat kf_depth_image;
    for( int i=0 ; i< (int) STATE["DataNodes"].size() ; i++ )
    {
        // auto data_node = STATE["DataNodes"][i];
        cout << "---\njson.seq = " << STATE["DataNodes"][i]["seq"] << endl;
        if( xloader.is_data_available(STATE["DataNodes"][i]) == false ) {
            cout << "no image or pose data, skip...\n";
            continue;
        }


        cv::Mat left_image, depth_image ;
        xloader.retrive_image_data_from_json_datanode( STATE["DataNodes"][i], left_image, depth_image );
        Matrix4d w_T_c;
        xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][i], w_T_c );


        cv::imshow( "win", left_image );
        print_usage();
        char ch = cv::waitKey( 0 );

        if( ch == 'k' )
        {
            cout << TermColor::iWHITE() << "Set this as keyframe\n" << TermColor::RESET() << endl;
            tracker.setKeyframe( left_image, 50 );

            #if 1
            assert( tracker.isKeyFrameSet() );
            cv::Mat dst;
            MiscUtils::plot_point_sets( tracker.keyframe_image(), tracker.keyframe_uv(), dst, cv::Scalar(255,0,0),true, "n_features="+to_string(tracker.keyframe_nfeatures()) );
            cv::imshow( "goodFeaturesToTrack", dst );
            #endif

            w_T_kf = w_T_c;
            kf_depth_image = depth_image;
            cout << MiscUtils::cvmat_info( kf_depth_image ) << endl;
        }

        if( ch == 'a' )
        {
            cout << TermColor::iWHITE() << "trackFromPrevframe\n" << TermColor::RESET() << endl;
            int successfully_tracked = tracker.trackFromPrevframe( left_image );

            #if 1
            assert( tracker.isKeyFrameSet() );
            cv::Mat dst;
            string msg ="n_features="+to_string(tracker.keyframe_nfeatures())+";successfully tracked = " + to_string(successfully_tracked) ;
            MiscUtils::plot_point_sets_masked( left_image,
                tracker.tracked_uv(), tracker.tracked_status(),
                dst, cv::Scalar(255,0,0),true, msg );
            cv::imshow( "tracked", dst );
            #endif


            #if 1
            // triangulate
            MatrixXd kf_uv = tracker.keyframe_uv();
            MatrixXd c_uv = tracker.tracked_uv();
            vector<bool> tracking_status = tracker.tracked_status();


            MatrixXd kf_normed_uv = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( xloader.left_camera, kf_uv );
            MatrixXd c_normed_uv  = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( xloader.left_camera, c_uv );

            VectorXd kf_z = StaticPointFeatureMatching::depth_at_image_coordinates( kf_uv, kf_depth_image );


            int ngood = 0;
            for( int h=0 ; h<kf_uv.cols() ; h++ ) // triangulate each tracked point
            {
                if( tracking_status[h] == false )
                    continue;

                double del_u = kf_uv(0,h) - c_uv(0,h);
                double del_v = kf_uv(1,h) - c_uv(1,h);
                cout << setprecision(4);
                cout << "h#" << setw(2) << h << "\t";
                cout << "del_u=" << setw(6) << abs( del_u ) << ", ";
                cout << "del_v=" << setw(6) << abs( del_v ) << ", ";
                cout << "parallax=" << setw(6) <<  sqrt( (del_u*del_u + del_v*del_v) ) << "\t";

                Vector4d result_X;
                Matrix4d c_T_kf = w_T_c.inverse() * w_T_kf;
                Triangulation::IterativeLinearLSTriangulation(
                    kf_normed_uv.col(h), Matrix4d::Identity(),
                    c_normed_uv.col(h), c_T_kf, result_X );

                cout << "\ttriangulated=" << result_X(2)<< "\t" ;
                cout << "depth_image=" << kf_z(h)  << "\t";

                double diff = abs( kf_z(h) - result_X(2) );
                if( diff < 0.2 ) { cout << TermColor::GREEN(); ngood++; }
                cout << "diff=" << diff << "\t";
                cout << TermColor::RESET();

                cout << endl;
            }
            cout << "ngood=" << ngood << endl;
            #endif

        }

        if( ch == 27 ) {
            cout << "QUIT....\n";
            break;
        }


        // cout << "any key to continue\n";
        // cv::waitKey(0);

    }

}
