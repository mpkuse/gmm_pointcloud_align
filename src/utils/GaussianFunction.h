#pragma once

// Static functions for evaluation of single variate and
// multivariate gaussians

#include <iostream>
#include <random>
#include <chrono>

using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;

class GaussianFunction
{
public:
    //---Single Variate
    // Eval the gaussian function at each x_i (i=1 to n) for the given mu, sigma.
    static VectorXd eval( VectorXd x, double mu, double sigma  );
    static double eval( double x, double mu, double sigma );
    static VectorXd linspace( double start_t, double end_t, int n );

    //---Multivariate


};
