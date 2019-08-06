#include "GaussianFunction.h"

VectorXd GaussianFunction::eval( VectorXd x, double mu, double sigma  )
{
    assert( x.rows() > 0 );
    // cout << "  allocate " << x.rows() << endl;
    VectorXd out = VectorXd::Zero( x.rows() );
    double factor = 1.0 / ( sigma * sqrt( 2.0 * M_PI ) );
    double sigma_sqr = sigma * sigma;

    for( int i=0 ; i<x.rows() ; i++ )
    {
        out(i) = exp( -1.0 / (2*sigma_sqr) * (x(i) - mu) * (x(i) - mu) );
    }
    out = factor * out;

    return out;
}

double GaussianFunction::eval( double x, double mu, double sigma )
{
    double factor = 1.0 / ( sigma * sqrt( 2.0 * M_PI ) );
    double result = factor * exp( -1.0 / (2*sigma*sigma) * (x - mu) * (x - mu) );
    return result;

}



VectorXd GaussianFunction::linspace( double start_t, double end_t, int n )
{
    assert( start_t < end_t  && n > 0 );
    // VectorXd t = ArrayXd::LinSpaced( start_t, end_t, n).matrix();
    VectorXd t = VectorXd::Zero( n );

    double v = start_t;
    double delta = (end_t - start_t) / double( n-1 );
    for( int i=0 ; i<n ; i++ )
    {
        // cout << "t(" << i << ") = " << v << endl;
        t(i) = v;
        v+= delta;
    }
    // cout << "t=\n" <<  t.transpose() << endl;
    return t;

}
