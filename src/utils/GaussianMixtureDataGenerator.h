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



class GaussianMixtureDataGenerator
{
public:
    GaussianMixtureDataGenerator( );
    GaussianMixtureDataGenerator( double seed  );


    /// Core 1D generators
    VectorXd gaussian_randoms( int n, double mu, double sigma );

    /// Generate 1d data with mixture of gaussians.
    VectorXd gaussian_mixtures_1d( vector<int> n, vector<double> mu, vector<double> sigma );


    // Multivariate random numbers gaussian distributed
    MatrixXd gaussian_multivariate_randoms( const int n, const VectorXd mu, const MatrixXd sigma );
private:
    std::mt19937 * generator;

};
