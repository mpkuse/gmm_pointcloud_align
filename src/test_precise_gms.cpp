// The issue with GMS matcher is that the point precision is often bad. This
// is causing issues with accuracy of reprojection error and hence the pose computation.
//
//      The idea here is to use GMS matcher's points as initial guess for matching
//      and to produce a sparser matches.
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/PointFeatureMatching.h"

#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

#include <Eigen/Sparse>

class TMP_Holder
{
public:
    // #define  ___StaticPointFeatureMatching__refine_and_sparsify_matches( msg ) msg;
    #define ___StaticPointFeatureMatching__refine_and_sparsify_matches( msg ) ;
    // Given a match refines it. Use this with GMS matcher which usually gives quite dense matching but not as precise.
    //      This does the following:
    //          a. Sparify the matches
    //          b. Do optical flow tracking using the matches as initial guess for LKOpticalFlow. This is done to improve precision of the point matches.
    //      Params:
    //      im_a, im_b : Input images (full resolution)
    //      uv_a, uv_b : Coarse matching points 2xN or 3xN. Initial matches to be refined for those 2 images respectively.
    //      refined_uv_a, refined_uv_b [output] : refined and sparsified points
    static void refine_and_sparsify_matches(
        const cv::Mat im_a, const cv::Mat im_b,
        const MatrixXd& uv_a, const MatrixXd& uv_b,
        MatrixXd& refined_uv_a, MatrixXd& refined_uv_b
    )
    {
        ___StaticPointFeatureMatching__refine_and_sparsify_matches(
        cout << TermColor::GREEN() << "[refine_and_sparsify_matches] starts\n" << TermColor::RESET(); )
        //---- safety when im_a is empty and uv_a is empty
        {
            if( im_a.empty() || im_b.empty() || uv_a.cols() == 0 || uv_b.cols() == 0 ) {
                cout << "[refine_and_sparsify_matches] either of the images of input points was empty\n";
                return;
            }
            if( uv_a.rows() == 2 || uv_a.rows() ==3 || uv_b.rows() == 2 || uv_b.rows() == 3 ) {
                ; // OK!
            }
            else {
                cout << "[refine_and_sparsify_matches] input uv_a,uv_b has to be 2xN or 3xN\n";
                return;
            }
        }


        // PARAMS
        int W = 31; //< 1 point max in a window of this half_win_size

        // END PARAMS


        //---- Sparsify pts
        // loop over all the point matches uv_a and note them in sparse matrix.
        // the purpose is to not pick too many points in a small neibhourhood
        MatrixXi mat = MatrixXi::Zero(im_a.rows, im_a.cols);
        vector<cv::Point2f> ret_uv_a, ret_uv_b;
        ___StaticPointFeatureMatching__refine_and_sparsify_matches( ElapsedTime t_sparsify("Sparsify"); )
        for( int i=0 ; i<uv_a.cols() ; i++ )
        {
            auto u = uv_a(0,i); //x coef
            auto v = uv_a(1,i); //y coef
            if( mat(v,u) > 0 ) {
                // cout << TermColor::RED() << u << "," << v << TermColor::RESET() <<  endl;
                continue;
            }

            auto ud = uv_b(0,i); //x coef
            auto vd = uv_b(1,i); //y coef

            // cout << u << "," << v << "<--->" << ud << "," << vd << endl;
            ret_uv_a.push_back( cv::Point2f(u,v) );
            ret_uv_b.push_back( cv::Point2f(ud,vd) );

            int v_m = v-int(W/2); //max( 0, v-int(W/2) );  //x coef
            int u_m = u-int(W/2); //max( 0, u-int(W/2) );  //y coef
            mat.block( v_m, u_m, W, W ) = MatrixXi::Constant( W,W, 1.0 );
            // for( int _u = -W ; _u<=W ; _u++)
            //     for( int _v = -W ; _v<=W ; _v++)
            //         mat.coeffRef(v+_v,u+_u) = true;
        }
        ___StaticPointFeatureMatching__refine_and_sparsify_matches(
        cout << t_sparsify.toc() << "\tn_retained=" << ret_uv_a.size() << " out of total=" << uv_a.cols() << endl; )



        //---- optical flow based refine
        ___StaticPointFeatureMatching__refine_and_sparsify_matches( ElapsedTime t_optflow( "Optical Flow Refinement");)
        vector<float> err;
        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);

        #if 0 // set this to 1 to use fwd and reverse optical flow, more robust
        vector<uchar> status1, status2;
        calcOpticalFlowPyrLK(im_a, im_b, ret_uv_a, ret_uv_b, status1, err, cv::Size(35,35), 4, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        calcOpticalFlowPyrLK(im_b, im_a, ret_uv_b, ret_uv_a, status2, err, cv::Size(35,35), 4, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        vector<uchar> status = MiscUtils::vector_of_uchar_AND( status1, status2 );
        ___StaticPointFeatureMatching__refine_and_sparsify_matches(
        cout << t_optflow.toc() << "\tOptical flow pts retained: (fwd,rev, AND)="
                    << "(" << MiscUtils::total_positives(status1 ) << ","
                    << MiscUtils::total_positives(status2 )  << ","
                    << MiscUtils::total_positives(status )
                    << ")" << endl;
                )
        #endif

        #if 1 //1 way optical flow
        vector<uchar> status;
        calcOpticalFlowPyrLK(im_a, im_b, ret_uv_a, ret_uv_b, status, err, cv::Size(35,35), 4, criteria, cv::OPTFLOW_USE_INITIAL_FLOW);
        ___StaticPointFeatureMatching__refine_and_sparsify_matches(
        cout <<  t_optflow.toc() << "\t optical flow retained points: " << MiscUtils::total_positives(status ) << endl; )
        #endif

        MiscUtils::reduce_vector( ret_uv_a, status );
        MiscUtils::reduce_vector( ret_uv_b, status );
        MiscUtils::point2f_2_eigen( ret_uv_a, refined_uv_a );
        MiscUtils::point2f_2_eigen( ret_uv_b, refined_uv_b );


        // for( int j=0 ; j<refined_uv_a.cols() ; j++ )
            // cout << refined_uv_a.col(j).transpose() << "<--->" <<refined_uv_b.col(j).transpose() << endl;

        ___StaticPointFeatureMatching__refine_and_sparsify_matches(
        cout << TermColor::GREEN() << "[refine_and_sparsify_matches] ends\n" << TermColor::RESET(); )
    }

};

int main_x( int __a__, int __b__);

int main()
{
    for( int iu=0 ;iu<10 ; iu++ )
    {
        int __a__ = rand() % 50;
        int __b__ = rand() % 50;
        main_x( __a__, __b__ );

    }
}

int main_x( int __a__, int __b__)
// int main( )
{
    cout << "Hello test_precise matching\n";


    // Load an Image Pair
    const string BASE = "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/";
    // const string fname_a = "odomSeq0_im_0.jpg";
    // const string fname_b =  "odomSeq1_im_0.jpg";
    const string fname_a = "odomSeq0_im_"+ to_string( __a__ ) +".jpg";
    const string fname_b =  "odomSeq1_im_"+ to_string( __b__ ) +".jpg";
    cv::Mat im_a = cv::imread( BASE+"/"+fname_a, 0 );
    cv::Mat im_b = cv::imread( BASE+"/"+fname_b, 0 );
    if( im_a.empty() || im_b.empty() )
    {
        cout << TermColor::RED() << "Cannot read input images" << TermColor::RESET();
        exit(1);
    }
    cout << "successfully loaded: " << fname_a << "\t" << fname_b << endl;


    // Low resolution GMS Matcher
    ElapsedTime t_gms( "GMS Matcher" );
    MatrixXd uv_a, uv_b;
    // StaticPointFeatureMatching::gms_point_feature_matches( im_a, im_b, uv_a, uv_b );
    StaticPointFeatureMatching::gms_point_feature_matches_scaled( im_a, im_b, uv_a, uv_b, 0.5 );
    // StaticPointFeatureMatching::gms_point_feature_matches_scaled( im_a, im_b, uv_a, uv_b, 0.25 );
    cout << t_gms.toc() << endl;

    if( uv_a.cols() == 0 )
        return 0;

    // plot
    cv::Mat dst;
    MiscUtils::plot_point_pair( im_a, uv_a, im_b, uv_b, dst, 3 );
    // MiscUtils::plot_point_pair( im_a, uv_a, im_b, uv_b, dst, cv::Scalar(0,0,255) );
    cv::imshow( "gms_point_feature_matches", dst );


    #if 1
    // Refine and Sparsify matching
    ElapsedTime t_refine_n_sparsify( "refine and sparsify matches" );
    MatrixXd refined_uv_a, refined_uv_b;
    TMP_Holder::refine_and_sparsify_matches(  im_a, im_b, uv_a, uv_b, refined_uv_a, refined_uv_b );
    cout << t_refine_n_sparsify.toc() << endl;
    #endif

    cv::Mat dst_refined;
    MiscUtils::plot_point_pair( im_a, refined_uv_a, im_b, refined_uv_b, dst_refined, cv::Scalar(0,0,255) );
    cv::imshow( "dst_refined", dst_refined );



    cv::waitKey(0);

}
