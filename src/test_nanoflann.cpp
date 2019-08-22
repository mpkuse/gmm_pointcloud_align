#include <iostream>
using namespace std;

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/nanoflann/nanoflann.hpp"

typedef nanoflann::KDTreeEigenMatrixAdaptor< MatrixXd >
  my_kd_tree_t;


int main()
{
    MatrixXd data = MatrixXd::Random( 3, 15 );

    my_kd_tree_t mat_index(3, std::cref(data.transpose()), 10 /* max leaf */);
    mat_index.index->buildIndex();

}
