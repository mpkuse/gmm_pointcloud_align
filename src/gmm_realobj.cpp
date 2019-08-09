#include <iostream>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "GMMFit.h"
#include "utils/GaussianFunction.h"
#include "utils/GaussianMixtureDataGenerator.h"
#include "utils/RosMarkerUtils.h"
#include "utils/MeshObject.h"


int main(int argc, char ** argv )
{
    // Ros init
    ros::init(argc, argv, "talker");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);


    //
    // Load OBJ

    MeshObject obj( "bunny.obj", 100.0 );

    MatrixXd w_X = obj.getVertices();
    cout << "vertices dims: " << w_X.rows() << "x" << w_X.cols() << endl;
    cout << w_X.leftCols(10) << endl;


    //
    // GMM

    // Initial Guess
    vector<VectorXd> init_mu;
    vector<MatrixXd> init_sigma;
    vector<visualization_msgs::Marker> __viz__;

    for(int i=0 ; i<25 ; i++ ) {
        int r = rand() % w_X.cols();
        VectorXd random_pt_on_obj = w_X.col(r).topRows(3);
        init_mu.push_back( random_pt_on_obj   );
        init_sigma.push_back( .1 * MatrixXd::Identity(3,3) );

    }


    #if 1
    // Before Fitting aka initial guess
    {
        for( int i=0 ; i<init_mu.size() ; i++ ) {
            visualization_msgs::Marker marker__gauss;
            RosMarkerUtils::init_mu_sigma_marker( marker__gauss, init_mu[i], init_sigma[i], 0.1 );
            RosMarkerUtils::setcolor_to_marker( 0.0, 1.0, 1.0, 1.0, marker__gauss );
            marker__gauss.id = i;
            marker__gauss.ns = "before";
            __viz__.push_back( marker__gauss );
        }
    }
    #endif



    // exec
    MatrixXd res = w_X.topRows(3);
    VectorXd priors;
    GMMFit::fit_multivariate( res, 25, init_mu, init_sigma, priors );



    {
        for( int i=0 ; i<init_mu.size() ; i++ ) {
            visualization_msgs::Marker marker__gauss;
            RosMarkerUtils::init_mu_sigma_marker( marker__gauss, init_mu[i], init_sigma[i], 0.1 );
            RosMarkerUtils::setcolor_to_marker( 0.0, 1.0, 0.0, 1.0, marker__gauss );
            marker__gauss.id = i;
            marker__gauss.ns = "after_"+to_string(i);
            __viz__.push_back( marker__gauss );
        }
    }


    // Generate randoms using the estimated means and gaussians
    cout << "GENERATE DATA WITH GAUSSIANS\n";
    vector<int> nx;
    GaussianMixtureDataGenerator gen(0);
    for( int i=0 ; i<init_mu.size() ; i++ ) {
        nx.push_back( int( priors(i)*2000 ) );
    }
    MatrixXd generated_res = gen.gaussian_mixtures_multivariate( nx, init_mu, init_sigma );
    cout << "Generated random data: res_" << res.rows() << "x" << res.cols() << endl;




    // Viz: a) Plot point cloud b) GMMs
    visualization_msgs::Marker marker;
    marker.ns = "objA";
    RosMarkerUtils::init_points_marker( marker );
    RosMarkerUtils::setcolor_to_marker( 1.0, 0.0, 0.0, 1.0, marker );
    RosMarkerUtils::setscaling_to_marker( 0.05, marker );
    for( int i=0 ; i<w_X.cols() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( w_X(0,i), w_X(1,i), w_X(2,i), marker, false );
    }

    visualization_msgs::Marker marker_generated;
    marker_generated.ns = "generated_data";
    RosMarkerUtils::init_points_marker( marker_generated );
    RosMarkerUtils::setcolor_to_marker( 1.0, 1.0, 0.0, 1.0, marker_generated );
    RosMarkerUtils::setscaling_to_marker( 0.05, marker_generated );
    for( int i=0 ; i<generated_res.cols() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( generated_res(0,i), generated_res(1,i), generated_res(2,i), marker_generated, false );
    }




    ros::Rate rate(10);
    while( ros::ok() )
    {
        rate.sleep();
        marker_pub.publish( marker  );
        marker_pub.publish( marker_generated  );

        for( int i=0 ; i<__viz__.size() ; i++ )
            marker_pub.publish( __viz__[i]  );

    }


    return 0;
}
