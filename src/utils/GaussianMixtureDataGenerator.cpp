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
