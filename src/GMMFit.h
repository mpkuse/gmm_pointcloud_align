#pragma once

// This class contains methods for fitting GMMs on data.
#include <iostream>
#include <vector>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/GaussianFunction.h"
#include "utils/TermColor.h"


class GMMFit
{
public:
    /// Fit a GMM Model with EM Algorithm
    /// Closely follows the process described here: https://www.youtube.com/playlist?list=PLBv09BD7ez_4e9LtmK626Evn1ion6ynrt
    ///     @param in_vec: The input vector
    ///     @param K : The number of gaussians
    ///     @param mu[input,output] : Initial guess of mu
    ///     @param sigma[input,output] : Initial guess of sigma
    static bool fit_1d( const VectorXd& in_vec, const int K, vector<double>& mu, vector<double>& sigma );
};
