// A simple mean of a bunch of numbers. However using huber norm.
// The advantage of this is that the mean will be robust to outliers
//     Need to do gradient decent. There is no closed form solution for this.




#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

double hubernorm_mean( VectorXd& V )
{
    const double HUBER_RANGE = .4;
    auto initial_guess = V.mean();
    // initial_guess = 10.0;

    cout << "initial_guess = " << initial_guess << endl;

    auto sum_L = 0.0;
    auto sum_deriv = 0.0;

    for( int newton_i=0 ; newton_i < 15 ; newton_i++ )
    {

        sum_L = 0.0;
        sum_deriv = 0.0;


        for( int i=0 ; i<V.cols() ; i++ ) // loop over all values
        {
            auto residue = initial_guess - V(i);

            auto L_i = ( abs(residue) < HUBER_RANGE )?( 0.5* residue*residue):( HUBER_RANGE*(abs(residue) - 0.5*HUBER_RANGE) );
            auto deriv_i = 0.0;
            if( abs(residue) < HUBER_RANGE ) {
                deriv_i = residue;
            } else if( residue > 0 ) {
                deriv_i = 1.0*HUBER_RANGE;
            } else if( residue < 0 ) {
                deriv_i = -1.0*HUBER_RANGE;
            } else assert( false);


            sum_L += L_i;
            sum_deriv += - deriv_i;
        }

        cout << "function=" << sum_L << "\tderiv=" << sum_deriv << endl;
        initial_guess -= 5.0/float(newton_i+1)  * sum_deriv;
        cout << "initial_guess ( after itr#" << newton_i << " ) = " << initial_guess << endl;
    }
}

double hubernorm_mean2( VectorXd& V )
{
    const double HUBER_RANGE = 0.4;
    auto initial_guess = V.mean();
    cout << "initial_guess = " << initial_guess << endl;


    auto sum_a = 0.0;
    auto sum_b = 0.0;
    for( int newton_i=0 ; newton_i<5 ; newton_i++ )
    {
        sum_a = 0; sum_b = 0;
        for (int p_i = 0; p_i < V.cols(); p_i++)
        {
            auto residual = initial_guess - V(p_i);
            if (residual < HUBER_RANGE && residual > -HUBER_RANGE)
            {
                sum_a += 2 * residual;
                sum_b += 2;
            }
            else
            {
                sum_a += residual > 0 ? HUBER_RANGE : -1 * HUBER_RANGE;
            }
            float delta_depth = -sum_a / (sum_b + 10.0);
            initial_guess = initial_guess + delta_depth;
            cout << "initial_guess ( after itr#" << newton_i << " ) = " << initial_guess << endl;
            if (delta_depth < 0.01 && delta_depth > -0.01)
                break;

        }
    }


}




int main()
{
    // DATA
    VectorXd data = VectorXd::Random( 20 ) * 10.;
    data(0) = 100.;

    cout << "data: " << data.transpose() << endl;
    cout << "data.mean() : " << data.mean() << endl;


    hubernorm_mean( data );
}
