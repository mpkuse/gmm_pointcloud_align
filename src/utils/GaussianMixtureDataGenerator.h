#pragma once

// This class helps generate gaussian distributed data.
// You could use it to generate random points disributed according to gaussian distribution.
// There are also functions to give you data points for gaussian mixtures. Also exist
// functions for shuffling the data

#include <iostream>
#include <random>
#include <chrono>

using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;

#include "GaussianFunction.h"

class GaussianMixtureDataGenerator
{
public:
    GaussianMixtureDataGenerator( );
    GaussianMixtureDataGenerator( double seed  );


    /// Core 1D generators
    VectorXd gaussian_randoms( int n, double mu, double sigma );

    /// Generate 1d data with mixture of gaussians.
    VectorXd gaussian_mixtures_1d( vector<int> n, vector<double> mu, vector<double> sigma );


    // Multivariate random numbers gaussian distributed.
    // will return a dxn matrix. d is the dimensions and n is number of datapoints
    // @param n: Number of random numbers to generate
    // @param mu : mean. dx1
    // @param sigma: covariance matrix dxd
    MatrixXd gaussian_multivariate_randoms( const int n, const VectorXd mu, const MatrixXd sigma );

    // Multivariate Gaussian mixture. Number of gaussians is 'l'
    // @param vector<int> n: number of datapoints in each of the 'l' gaussian mixture. n.size() == l
    // @param vector<VectorXd> : means of each of the l gaussians. Dimension (d) of gassuian is determined from size of VectorXd
    // @param vector<MatrixXd> : covariance matrix of each gaussian. Square symmetric matrix. with dimensions same as mu
    // @param return 3xn matrix
    MatrixXd gaussian_mixtures_multivariate( const vector<int> n, vector<VectorXd> mu, vector<MatrixXd> sigma );
private:
    std::mt19937 * generator;

};
