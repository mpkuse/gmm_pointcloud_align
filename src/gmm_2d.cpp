// fit 2d gaussians

#include <iostream>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "utils/RosMarkerUtils.h"

#include "utils/GaussianMixtureDataGenerator.h"
#include "utils/GaussianFunction.h"
#include "GMMFit.h"




int main( int argc, char ** argv )
{
    // Ros init
    ros::init(argc, argv, "talker");
    ros::NodeHandle nh;


    // Publisher
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);


    #if 0
    // ----> Single Multivariate Gaussian
    // mu and sigma
    MatrixXd r = MatrixXd::Random(2,2);
    MatrixXd sigma = r + r.transpose();
    // MatrixXd sigma = MatrixXd::Identity(2,2);
    sigma(0,0) *= 5;
    sigma(0,1) = 2.83;
    sigma(1,0) = 2.83;
    cout << TermColor::GREEN() << "sigma=\n" << sigma << TermColor::RESET() << endl;
    cout << "sigma.determinant()=" << sigma.determinant() << endl;


    VectorXd mu = VectorXd::Random(2);
    mu(0) = 10.0;
    cout <<TermColor::GREEN() << "mu=" << mu.transpose() << TermColor::RESET() << endl;


    GaussianMixtureDataGenerator gen(0);
    MatrixXd res = gen.gaussian_multivariate_randoms( 500, mu, sigma );
    cout << "res_" << res.rows() << "x" << res.cols() << endl;
    // cout << res << endl;

    // compute sample mean and sample covariance to make sure the characteristics were as desired
    cout << "sample mean: " << GaussianFunction::sample_mean( res ).transpose() << endl;
    cout << "sample cov mat:\n" << GaussianFunction::sample_covariance_matrix( res ) << endl;
    #endif



    // ----> Mixture of Multivariate Gaussian
    #if 1
    vector<int> nx;
    vector<VectorXd> mu;
    vector<MatrixXd> sigma;
    for( int l=0 ; l<2 ; l++ ) //number of gaussians
    {
        MatrixXd r = MatrixXd::Random(2,2);
        MatrixXd _sigma = r + r.transpose();
        _sigma(0,0) = abs( _sigma(0,0) );
        _sigma(1,1) = abs( _sigma(1,1) );
        // MatrixXd _sigma = MatrixXd::Identity(2,2);

        VectorXd _mu = VectorXd::Random(2)  ;


        if( l==1 ) { _mu *= 50. ;  _sigma(1,1) *= 15.;  _sigma(0,1) = 1.1; _sigma(1,0) = 1.1; }

        cout << TermColor::GREEN() << "#" << l << "\tn=" << 100 << "\tmu=" << _mu.transpose() << TermColor::RESET() << endl;
        cout << "sigma:\n" << _sigma << "\n"<< endl;

        GaussianFunction::isValidCovarianceMatrix( _sigma );

        nx.push_back( 100 );
        sigma.push_back( _sigma );
        mu.push_back( _mu );
    }

    GaussianMixtureDataGenerator gen(0);
    MatrixXd res = gen.gaussian_mixtures_multivariate( nx, mu, sigma );
    cout << "res_" << res.rows() << "x" << res.cols() << endl;
    #endif




    // Do GMM Fit 




    // xy plot
    visualization_msgs::Marker marker;
    marker.ns = "randoms";
    RosMarkerUtils::init_points_marker( marker );
    RosMarkerUtils::setcolor_to_marker( 1.0, 0.0, 0.0, 1.0, marker );
    RosMarkerUtils::setscaling_to_marker( 0.1, marker );
    for( int i=0 ; i<res.cols() ; i++ )    {
        RosMarkerUtils::add_point_to_marker( res(0,i), res(1,i), 0.0, marker, false );
    }


    ros::Rate rate(10);
    while( ros::ok() )
    {
        rate.sleep();
        marker_pub.publish( marker  );
    }
    return 0;

}
