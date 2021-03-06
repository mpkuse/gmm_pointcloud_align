#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

using namespace std;

#include "ElapsedTime.h"

#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>


class MiscUtils
{
public:
    //---------------------------- INFO ---------------------------------//

    static string type2str(int type);
    static string cvmat_info( const cv::Mat& mat );
    static string cvmat_minmax_info( const cv::Mat& mat );
    static string imgmsg_info(const sensor_msgs::ImageConstPtr &img_msg);
    static string imgmsg_info(const sensor_msgs::Image& img_msg);
    static cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg);

    static std::vector<std::string>
    split( std::string const& original, char separator );


    //---------------------------- Conversions ---------------------------------//
    // convert from opencv format of keypoints to Eigen
    static void keypoint_2_eigen( const std::vector<cv::KeyPoint>& kp, MatrixXd& uv, bool make_homogeneous=true );

    // given opencv keypoints and DMatch will produce M1, and M2 the co-ordinates
    static void dmatch_2_eigen( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                const std::vector<cv::DMatch> matches,
                                MatrixXd& M1, MatrixXd& M2,
                                bool make_homogeneous=true
                            );

    // given a vector of Point2f will return a 2xN or 3xN matrix (if make_homogeneous=true).
    static void point2f_2_eigen( const std::vector<cv::Point2f>& p, MatrixXd& dst, bool make_homogeneous=true );

    // given an input eigen matrix of eithen 2xN or 3xN make this into vector<Point2f>.
    static void eigen_2_point2f( const MatrixXd& inp, std::vector<cv::Point2f>& p );

    // given a vector of Point3f will return a 3xN or 4xN matrix (if make_homogeneous=true).
    static void point3f_2_eigen( const std::vector<cv::Point3f>& p, MatrixXd& dst, bool make_homogeneous=true );

    // given an input eigen matrix of eithen 2xN or 3xN make this into vector<Point2f>.
    static void eigen_2_point3f( const MatrixXd& inp, std::vector<cv::Point3f>& p );


    // Given multiple point sets `mats` each of sizes 3xN1, 3xN2, .... 3xNn and corresponding valids gather everything into dst
    // eg. say the valids look  like [ [11101], [000101111], [11111111100] ] will return only 3d points with valids as 1
    static void gather( const vector<MatrixXd>& mats, const vector<  vector<bool> >& valids, MatrixXd& dst );

    // Given multiple mats (in the vector) each of size 3xN, 3xN2, ... (cols need to be same for each item)
    static void gather( const vector<MatrixXd>& mats, MatrixXd& dst );

    // N1x1, N2x1, .... (each VectorXd can be of different sizes, will be conatenated as a big vector)
    static void gather( const vector<VectorXd>& mats, VectorXd& dst );

    static int total_true( const vector<bool>& V );
    static int total_positives( const vector<uchar>& V );
    
    static vector<bool> filter_near_far( const VectorXd& dd, double near, double far );
    static vector<bool> vector_of_bool_AND( const vector<bool>& A, const vector<bool>& B );
    static vector<uchar> vector_of_uchar_AND( const vector<uchar>& A, const vector<uchar>& B );


    static VectorXd to_eigen( const vector<uchar>& V );


    static void reduce_vector(vector<cv::Point2f> &v, const vector<uchar> status); //inplace
    static void reduce_vector(const vector<cv::Point2f> &v, const vector<uchar> status, vector<cv::Point2f>& out );

    //---------------------------- Conversions ---------------------------------//


    //-------------------------------- IMSHOW ----------------------------------//
    static void imshow( const string& win_name, const cv::Mat& m, float scale=1.0 );


    //--------------------- Plot Keypoints on Image ----------------------------//
    // Eigen Interace:  PLotting functions with Eigen Interfaces
    static void plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                            const cv::Scalar& color, bool enable_keypoint_annotation=true,const string& msg=string("") );

    // cv::Mat Interfaces: Plotting Functions with cv::Mat Interfaces.
    static void plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst,
                                            const cv::Scalar& color, bool enable_keypoint_annotation=true, const string& msg=string("") );

    // Inplace plotting. Here dont need to specify a separate destination. src is modified.
    static void plot_point_sets( cv::Mat& im, const MatrixXd& pts_set,
                                            const cv::Scalar& color, bool enable_keypoint_annotation=true, const string& msg=string("") );

    // Plotting with annotations specified by VectorXi
    static void plot_point_sets( cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                            const cv::Scalar& color, const VectorXi& annotations, const string& msg );

    // Plotting with annotations specified by VectorXi inplace
    static void plot_point_sets( cv::Mat& im, const MatrixXd& pts_set,
                                            const cv::Scalar& color, const VectorXi& annotations, const string& msg );

    // plot point with colors specified at every point. pts_set : 3xN or 2xN, len(color_annotations) == pts_set.cols()
    static void plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                            vector<cv::Scalar>& color_annotations, float alpha=0.8, const string& msg=string("N/A") );



    // plot point set on image.
    //  im : Input image
    //  pts_set : 2xN or 3xN matrix with x,y in a col, in terms of image row and colidx this will be c,r.
    //  status : same size as im, once with status[k] == false will not be plotted
    //  dst [output]: output image
    static void plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set, const vector<uchar>& status,
            cv::Mat& dst,
            const cv::Scalar& color, bool enable_keypoint_annotation = true, const string msg = "" );
    static void plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set, const vector<bool>& status,
            cv::Mat& dst,
            const cv::Scalar& color, bool enable_keypoint_annotation = true, const string msg = "" );

    static void plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set,
            const VectorXd& status, double show_only_greater_than_this_value,
            cv::Mat& dst,
            const cv::Scalar& color, bool enable_keypoint_annotation = true, const string msg = "" );



    // Given an image and the mask (dimension same as the input image).
    //      dst( i,j ) = color if mask(i,j) > 0
    //      dst(i,j)   = untouched if mask(i,j) == 0
    static void mask_overlay( const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, cv::Scalar color );
    static void mask_overlay( cv::Mat& src, const cv::Mat& mask, cv::Scalar color );
    // END--------------------- Plot Keypoints on Image ----------------------------//




    //------------------------------- Plot Matchings on image pair -------------------------//

    // Plots [ imA | imaB ] with points correspondences
    // [Input]
    //    imA, imB : Images
    //    ptsA, ptsB : 2xN or 3xN
    //    idxA, idxB : Index of each of the image. This will appear in status part. No other imppact of these.
    //    color_marker : color of the point marker
    //    color_line   : color of the line
    //    annotate_pts : true with putText for each point. False will not putText.
    // [Output]
    //    outImg : Output image
    static void plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                          const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                          cv::Mat& dst,
                          const cv::Scalar& color_marker,
                          const cv::Scalar& color_line=cv::Scalar(0,255,0),
                          bool annotate_pts=false,
                          const string& msg=string("N.A")
                         );

     static void plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA,
                           const cv::Mat& imB, const MatrixXd& ptsB,
                           cv::Mat& dst,
                           const cv::Scalar& color_marker,
                           const string& msg=string("N.A"),
                           const cv::Scalar& color_line=cv::Scalar(0,255,0),
                           bool annotate_pts=false
                          );


     // nearly same as the above, but will color every co-ordinate with different color
     // color_map_direction : 0 ==> // horizontal-gradiant
     //                       1 ==>  // vertical-gradiant
     //                       2 ==> // manhattan-gradiant
     //                       3 ==> // image centered manhattan-gradiant
     static void plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                           const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                           cv::Mat& dst,
                           short color_map_direction,
                           const string& msg=string("N.A")
                          );

      static void plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA,
                            const cv::Mat& imB, const MatrixXd& ptsB,
                            cv::Mat& dst,
                            short color_map_direction,
                            const string& msg=string("N.A")
                           );

    //------------------------------- Plot Matchings on image pair -------------------------//


    //------------------------- Points and Lines on Images --------------------------------//

    // Given two image-points draw line between them, extend both ways. Infinite line-segments
    static void draw_fullLine(cv::Mat& img, cv::Point2f a, cv::Point2f b, cv::Scalar color);

    // draw line on the image, given a line equation in homogeneous co-ordinates. l = (a,b,c) for ax+by+c = 0
    static void draw_line( const Vector3d l, cv::Mat& im, cv::Scalar color );

    // mark point on the image, pt is in homogeneous co-ordinate.
    static void draw_point( const Vector3d pt, cv::Mat& im, cv::Scalar color  );

    // mark point on image
    static void draw_point( const Vector2d pt, cv::Mat& im, cv::Scalar color  );


    // append a status image . ';' separated
    static void append_status_image( cv::Mat& im, const string& msg, float txt_size=0.4, cv::Scalar bg_color=cv::Scalar(0,0,0), cv::Scalar txt_color=cv::Scalar(255,255,255), float line_thinkness=1.5 );
    static bool side_by_side( const cv::Mat& A, const cv::Mat& B, cv::Mat& dst );
    static bool vertical_side_by_side( const cv::Mat& A, const cv::Mat& B, cv::Mat& dst );

    // END ------------------------- Points and Lines on Images --------------------------------//


    // [Input] : f a float between 0 and 1.
    // [Output]: cv::Scalar gives out a bgr color pallet.
    // Note: This is inefficient, don't use it too often. If you are going to do lot of quering for colors use `class FalseColors`
    static cv::Scalar getFalseColor( float f );


private:

    static double Slope(int x0, int y0, int x1, int y1);

};


class FalseColors
{
public:
    FalseColors() {
        cv::Mat colormap_gray = cv::Mat::zeros( 1, 256, CV_8UC1 );
        for( int i=0 ; i<256; i++ ) colormap_gray.at<uchar>(0,i) = i;
        cv::applyColorMap(colormap_gray, colormap_color, cv::COLORMAP_JET	);
    }

    // f need to be between 0 and 1
    cv::Scalar getFalseColor( float f ) {
        int idx = (int) (f*255.);
        if( f<0 ) {
            idx=0;
        }
        if( f>1.0 ) {
            idx=255;
        }


        cv::Vec3b f_ = colormap_color.at<cv::Vec3b>(0,  (int)idx );
        cv::Scalar color_marker = cv::Scalar(f_[0],f_[1],f_[2]);
        return color_marker;
    }

    cv::Mat getStrip( int nrows, int ncols ) {
        cv::Mat colormap_gray = cv::Mat::zeros( nrows, ncols, CV_8UC1 );

        for( int r=0; r<nrows; r++ ) {
            for( int c=0 ; c<ncols; c++ )
                colormap_gray.at<uchar>(r,c) = (uchar) ( (float(c)/ncols)*256 );
        }

        cv::Mat __dst;
        cv::applyColorMap(colormap_gray, __dst, cv::COLORMAP_JET	);
        return __dst;
    }


    static cv::Scalar randomColor( int rng )
    {
      //   int icolor = (unsigned) rng;

      #if 0
      srand( rng );
        int icolor = (unsigned) rand();
        //TODO: to get even better colors use rng to generave a hue. With max saturation and brightness convert it to RGB
        return cv::Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
       #endif



       cv::Mat ze = cv::Mat::zeros( 3,3, CV_8UC3 );
      //  ze.at<cv::Vec3b>(0,0)[0] = rand() % 255;
       ze.at<cv::Vec3b>(0,0)[0] = int( 5458. / (rng+1. ) ) % 255;
       ze.at<cv::Vec3b>(0,0)[1] = 255;
       ze.at<cv::Vec3b>(0,0)[2] = 255;

       cv::Mat rgb;
       cv::cvtColor(ze, rgb,cv::COLOR_HSV2BGR);
       int r, g, b;
       r = rgb.at<cv::Vec3b>(0,0)[0];
       g = rgb.at<cv::Vec3b>(0,0)[1];
       b = rgb.at<cv::Vec3b>(0,0)[2];
       return cv::Scalar( r,g,b );




    }

private:
    cv::Mat colormap_color;


};
