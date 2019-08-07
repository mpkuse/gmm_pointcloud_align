#include "GaussianMixtureDataGenerator.h"

GaussianMixtureDataGenerator::GaussianMixtureDataGenerator(  )
{
    // cout << "Default contructor, so timebased random seed\n";
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator=new std::mt19937( seed );
}


GaussianMixtureDataGenerator::GaussianMixtureDataGenerator( double seed )
{
    // cout << "Seed with seed=" << seed << endl;
    generator=new std::mt19937( seed );
}



//----------------------------------- RANDOMS 1D---------------------------------//

VectorXd GaussianMixtureDataGenerator::gaussian_randoms( int n, double mu, double sigma )
{
    assert( n>0 );
    auto distribution = std::normal_distribution<double>(  mu, sigma );
    VectorXd vec = VectorXd::Zero( n );

    for( int i=0 ; i<n ; i++ )
    {
        double number = distribution(*generator);
        vec( i ) = number;
        // cout << "vec(" << i << ") := " << number << endl;
    }

    return vec;
}


/// Generate 1d data with mixture of gaussians.
VectorXd GaussianMixtureDataGenerator::gaussian_mixtures_1d
        ( vector<int> n, vector<double> mu, vector<double> sigma )
{
    int K = n.size();
    assert( K > 0 );
    assert( mu.size() == K && sigma.size() == K );

    // see how many number we are request and allocate
    int total_ns = 0;
    for( int k=0 ; k<K ; k++ )
    {
        assert( n[k] > 0 );
        total_ns += n[k];
    }
    // cout << "K= " << K << "  Allocate vector size: " << total_ns << endl;
    VectorXd out = VectorXd::Zero( total_ns );



    int s = 0;
    for( int k=0 ; k<K ; k++ )
    {
        // cout << "#" << k << "  gaussian_randoms( n=" << n[k] << ", mu=" << mu[k] << ", sigma=" << sigma[k] << ")" << endl;
        VectorXd tmp = gaussian_randoms( n[k], mu[k], sigma[k] );

        // cout << "out.segment( " << s << ", " << n[k] << ") = tmp\n";
        out.segment( s, n[k] ) = tmp;
        s+= n[k];
    }

    return out;
}


//------------------------Multivariate ------------------------------------//

MatrixXd GaussianMixtureDataGenerator::gaussian_multivariate_randoms(
    const int n, const VectorXd mu, const MatrixXd sigma )
{
    int d = mu.rows();
    //
    // Assert
    assert( n > 0 );
    assert( d>0 && mu.rows() == d );
    assert( sigma.rows() == sigma.cols() && sigma.rows() == d );

    assert( GaussianFunction::isValidCovarianceMatrix(sigma) );
    MatrixXd out = MatrixXd::Zero( d,n );


    //
    // Generate n dimensional normally distributed
    for( int i=0 ; i<d ; i++ )
    {
        out.row(i) = gaussian_randoms( n, 0.0, 1.0 );
    }


    //
    // Shape the covariance as desired
    //  -- Eigen values and Eigen values
    //  -- construct transform
    //  -- do the transform on out
    EigenSolver<MatrixXd> es(sigma, true);
    VectorXcd eigs = es.eigenvalues();
    MatrixXcd eig_vec = es.eigenvectors();
    // cout << "eigs: " << eigs << endl;
    // cout << "eig_vec: " << eig_vec << endl;
    // cout << "eigs.real().cwiseSqrt(): " << eigs.real().cwiseSqrt() << endl;
    MatrixXd T = eig_vec.real() * eigs.real().cwiseSqrt().asDiagonal() ;  // dxd matrix
    // cout << "T:" << T << endl;


    return  ( T * out ).colwise() + mu;


}


MatrixXd GaussianMixtureDataGenerator::gaussian_mixtures_multivariate(
    const vector<int> n, vector<VectorXd> mu, vector<MatrixXd> sigma )
{
    //
    // Verify if inputs are ok.
    int L = n.size(); //number of gaussians
    assert( L > 0 && mu.size() == L && sigma.size() == L );

    int D = mu[0].rows(); //dimension of the gaussians
    assert( D> 1 );
    int n_total_pts = 0 ;
    for( int i=0 ; i<L ; i++ ) {
        assert( n[i] > 0 );
        n_total_pts += n[i];
        assert( mu[i].rows() == D );
        assert( sigma[i].rows() == D && sigma[i].cols() == D );
        assert( GaussianFunction::isValidCovarianceMatrix( sigma[i]) );
    }


    //
    // Generate each of the gaussians
    MatrixXd out = MatrixXd::Zero( D, n_total_pts );
    int s = 0;
    for( int l=0 ; l<L ; l++ )
    {
        MatrixXd _tmp = gaussian_multivariate_randoms( n[l], mu[l], sigma[l] );
        assert( _tmp.rows() == D && _tmp.cols() == n[l] );

        out.block( 0, s,  D, n[l] ) = _tmp;
        s += n[l];

    }


    return out;

}
