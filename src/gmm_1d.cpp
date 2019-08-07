// Here I generate a 1D dataset with 2 gaussian distributions.
// Objective is to fit a GMM on this data with K=2.
// Also verify correctness by plotting with rviz

#include <iostream>
#include <random>

using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "utils/RosMarkerUtils.h"

#include "utils/GaussianMixtureDataGenerator.h"
#include "utils/GaussianFunction.h"
#include "GMMFit.h"

int main( int argc, char ** argv)
{
    // Ros init
    ros::init(argc, argv, "talker");
    ros::NodeHandle nh;


    // Publisher
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);



    // Data Generator
    GaussianMixtureDataGenerator gen( 0); //fixed seed
    // GaussianMixtureDataGenerator gen = GaussianMixtureDataGenerator( ); //time based seed

    #if 0
    // Gaussian randoms
    VectorXd g_1d = gen.gaussian_randoms( 1000, 16, 4 );
    cout << g_1d << endl;

    cout << "sample mean: " << GaussianFunction::sample_mean( g_1d ) << endl;
    cout << "sample var: " << GaussianFunction::sample_variance( g_1d ) << endl;
    return 0;
    #endif

    #if 1
    // Mix of Gaussians randoms
    vector<int> n = {200,600};
    vector<double> mu = {5.1, 12.5};
    vector<double> sigma = {8.0, 8.0};
    VectorXd g = gen.gaussian_mixtures_1d( n, mu, sigma );
    cout << "g:" << g.transpose() << endl;
    #endif




    #if 1
    // Fit 1d gmm
    vector<double> mu_s = { 0.0, 10.0};
    vector<double> sigma_s = {5.0, 5.0};
    GMMFit::fit_1d( g, 2, mu_s, sigma_s );
    #endif


    // try plotting as visualization_msgs::Marker
    //--- the random data points
    visualization_msgs::Marker marker;
    marker.ns = "randoms";
    RosMarkerUtils::init_points_marker( marker );
    RosMarkerUtils::setcolor_to_marker( 1.0, 0.0, 0.0, 1.0, marker );
    RosMarkerUtils::setscaling_to_marker( 0.1, marker );
    for( int i=0 ; i<g.rows() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( g(i), drand48(), 0.0, marker, false );
    }


    //--- plots
    VectorXd lin_space = GaussianFunction::linspace( -100, 100, 500 );
    VectorXd lin_gauss0 = GaussianFunction::eval( lin_space, mu_s[0], sigma_s[0] );
    VectorXd lin_gauss1 = GaussianFunction::eval( lin_space, mu_s[1], sigma_s[1] );

    visualization_msgs::Marker plot0;
    plot0.ns = "plot0";
    RosMarkerUtils::init_line_strip_marker( plot0 );
    RosMarkerUtils::setcolor_to_marker( 0.0, 1.0, 0.0, 1.0, plot0 );
    // RosMarkerUtils::setscaling_to_marker( 0.1, marker );
    for( int i=0 ; i<lin_space.rows() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( lin_space(i), 100.*lin_gauss0(i), 0.0, plot0, false );
    }

    visualization_msgs::Marker plot1;
    plot1.ns = "plot1";
    RosMarkerUtils::init_line_strip_marker( plot1 );
    RosMarkerUtils::setcolor_to_marker( 0.0, 1.0, 1.0, 1.0, plot1 );
    // RosMarkerUtils::setscaling_to_marker( 0.1, marker );
    for( int i=0 ; i<lin_space.rows() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( lin_space(i), 100.*lin_gauss1(i), 0.0, plot1, false );
    }


    ros::Rate rate(10);
    while( ros::ok() )
    {
        rate.sleep();
        marker_pub.publish( marker  );
        marker_pub.publish( plot0  );
        marker_pub.publish( plot1  );

    }


    return 0;
}
