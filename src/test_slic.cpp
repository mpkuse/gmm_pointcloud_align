// A standalone testing for SLIC

#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"


//
#include "SlicClustering.h"

#include "utils/MiscUtils.h"
#include "utils/ElapsedTime.h"
#include "utils/TermColor.h"

const string pkg_path = "/app/catkin_ws/src/gmm_pointcloud_align/";

void load_data( cv::Mat& left_image, cv::Mat& right_image, cv::Mat& depth_map )
{
    string opencv_file_name = pkg_path+"/resources/rgbd_samples/test.xml";
    cout << "Open File: " << opencv_file_name << endl;
    cv::FileStorage opencv_file(opencv_file_name, cv::FileStorage::READ);
    if( opencv_file.isOpened() == false ) {
        cout << "ERROR....><><> Cannot open opencv_file: " << opencv_file_name << endl;
        exit(1);
    }
    cout << "...OK!\n";
    opencv_file["left_image"] >> left_image;
    opencv_file["right_image"] >> right_image;
    opencv_file["depth_map"] >> depth_map;
    opencv_file.release();

    cout << "left_image\t" << MiscUtils::cvmat_info( left_image ) << endl;
    cout << "right_image\t" << MiscUtils::cvmat_info( right_image ) << endl;
    cout << "depth_map\t" << MiscUtils::cvmat_info( depth_map ) << endl;
}
int main()
{
    ElapsedTime t_slic;
    cv::Mat left_image, right_image, depth_map;
    load_data(left_image, right_image, depth_map);

    cv::imshow( "left_image", left_image );
    cv::waitKey(0);


    //------------SLIC Usage
    int w = left_image.cols, h = left_image.rows;
    int nr_superpixels = 400;
    double step = sqrt((w * h) / ( (double) nr_superpixels ) ); ///< step size per cluster
    cout << "===SLIC Params:\n";
    cout << "step size per cluster: " << step << endl;
    cout << "Number of superpixel: "<< nr_superpixels << endl;
    cout << "===\n";


    auto slic_obj = SlicClustering();

    t_slic.tic();
    slic_obj.generate_superpixels( left_image, depth_map, step );
    cout << TermColor::BLUE() << "t_slic (ms): " << t_slic.toc_milli() << TermColor::RESET() << endl;


    // slic_obj.display_center_grid( left_image, cv::Scalar(0,0,255) );
    cv::Mat output;
    slic_obj.display_contours( left_image, cv::Scalar(0,0,255), output );
    slic_obj.display_center_grid( output, cv::Scalar(0,255,0) );


    #if 1
    std::string calib_file = pkg_path+"/resources/realsense_d435i_left.yaml";
    cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
    camodocal::CameraPtr  m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    std::cout << ((m_camera)?"m_camera cam is initialized":"m_camera is not initiazed") << std::endl; //this should print 'initialized'


    // check the reprojection of 3d points
    MatrixXd c_V = slic_obj.retrive_superpixel_XYZ( true );

    if( !m_camera ) {
        cout << TermColor::RED() << "[mains] FATAL ERROR The cameras was not set...you need to set the camera to this class before calling the perspective function\n" << TermColor::RESET();
        exit(2);
    }

    MatrixXd c_v = MatrixXd::Zero( 3, c_V.cols() );
    for( int i=0 ; i<c_V.cols() ; i++ ) {
        Vector2d p;
        m_camera->spaceToPlane( c_V.col(i).topRows(3), p );
        c_v.col(i) << p, 1.0;
    }


    MiscUtils::plot_point_sets( output, c_v, output, cv::Scalar(255,0,0), false, "reprojected pts in blue" );

    #endif


    cv::imshow( "display-poutput", output );
    cv::waitKey(0);


}
