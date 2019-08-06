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

int main()
{
    //---- verify isValidCovarianceMatrix
    MatrixXd r = MatrixXd::Random(2,2);
    MatrixXd sigma = r + r.transpose();
    // MatrixXd sigma = MatrixXd::Identity(2,2);
    // sigma << 1.0, -.50, 0.0,
    //         0.0 , 1.0, 4.0,
    //         0.0,  0.0, 1.0;
    cout << "sigma_" << sigma.rows() << "x" << sigma.cols() << endl;
    cout << sigma << endl;

    // Eigen Values and singular values
    EigenSolver<MatrixXd> es(sigma, true);
    cout << "Eigen values of sigma=" << es.eigenvalues().transpose() << endl;
    JacobiSVD<MatrixXd> svd( sigma, ComputeFullV | ComputeFullU );
    cout << "Singular values of sigma= " << svd.singularValues().transpose() << endl;


    cout << "isValidCovarianceMatrix: " << GaussianFunction::isValidCovarianceMatrix( sigma ) << endl;
    cout << "sigma.determinant() = " << sigma.determinant() << endl;


    // TODO: construct positive definite matrix

    //
    VectorXd mu = VectorXd::Random(2);
    cout << "mu=" << mu.transpose() << endl;
    //
    cout << "gauss( X ) = " << GaussianFunction::eval( mu, mu, sigma ) << endl;;


    cout << TermColor::iRED() << "---------" << TermColor::RESET() << endl;
    MatrixXd x = MatrixXd::Random(2, 10);
    VectorXd res1 = GaussianFunction::eval( x, mu, sigma );
    for( int i=0 ; i<10 ; i++ )
    {
        VectorXd x_i = x.col(i);
        cout << GaussianFunction::eval( x_i, mu, sigma ) << "\t" << res1(i) << endl;
    }


    cout << TermColor::iRED() << "---------" << TermColor::RESET() << endl;
    // TODO Random numbers a) With uncorelated x, y
    //                     b) with corelated x,y
    // TODO Plot this on rviz

}
