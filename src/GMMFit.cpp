#include "GMMFit.h"

bool GMMFit::fit_1d( const VectorXd& in_vec, const int K, vector<double>& mu, vector<double>& sigma )
{
    assert( K>0 );
    assert( K == mu.size() && K==sigma.size() );

    cout << TermColor::GREEN() << "====GMMFit::fit_1d====\n" << TermColor::RESET();

    cout << "Initial Guess:\n";
    for( int k=0 ; k<K ; k++ ) {
        cout << "#" << k << "\tmu=" << mu[k] << "\tsigma=" << sigma[k] << endl;
    }

    // Priors (len will be K), 1 prior per class
    VectorXd priors = VectorXd::Zero(K);
    for( int k=0 ; k<K ; k++) priors(k) = 1.0/double(K);
    cout << "initial priors: " << priors.transpose() << endl;



    for( int itr=0 ; itr<10 ; itr++ ) {
        cout << "---itr=" << itr << endl;

        // Compute Likelihoods ie. P( x_i / b_k ) \forall i=1 to n , \forall k=1 to K
        cout << TermColor::YELLOW() << "Likelihood Computation" << TermColor::RESET() << endl;
        MatrixXd L = MatrixXd::Zero( in_vec.rows(), K );
        for( int k=0 ; k<K; k++ )
        {
            VectorXd _tmp = GaussianFunction::eval( in_vec, mu[k], sigma[k] );
            L.col(k) = _tmp;
        }
        // cout << "Likelihood for each datapoint (rows); for each class (cols):\n" << L << endl;



        // Posterior
        cout << TermColor::YELLOW() << "Posterior Computation (Bayes Rule)" << TermColor::RESET() << endl;
        MatrixXd P = MatrixXd::Zero( in_vec.rows() , K );
        for( int i=0 ; i<in_vec.rows() ; i++ ) //loop over datapoints
        {
            // cout << "L.row("<< i << " ) " << L.row(i) * priors << endl;
            double p_x = L.row(i) * priors; // total probability (denominator in Bayes rule)

            P.row(i) = ( L.row(i).transpose().array() * priors.array() ).matrix() / p_x;
        }
        // cout << "Posterior:\n" << P << endl;



        // Update Mu, Sigma
        cout << TermColor::YELLOW() << "Update mu, sigma" << TermColor::RESET() << endl;
        for( int k=0 ; k<K ; k++ )
        {
            double denom = P.col(k).sum();

            // Updated mu
            double mu_new_numerator = P.col(k).transpose() * in_vec  ;
            double mu_new = mu_new_numerator / denom;

            // updated sigma
            VectorXd _tmp = (  ( in_vec.array() - mu_new )  * ( in_vec.array() - mu_new )    ).matrix();
            double sigma_new_numerator = P.col(k).transpose()  * _tmp;

            double sigma_new = sqrt( sigma_new_numerator / denom );

            cout << "mu_new_"<< k << " " << mu_new << "\t";
            cout << "sigma_new_"<< k << " " << sigma_new << endl;
            mu[k] = mu_new;
            sigma[k] = sigma_new;
        }


        // update priors
        cout << TermColor::YELLOW() << "Update priors" << TermColor::RESET() << endl;
        for( int k=0 ; k<K ; k++ ) {
            priors(k) = ( P.col(k).sum() / double(in_vec.size()) );
        }
        cout << "updated priors: " << priors.transpose() << endl;


    }


    cout << TermColor::GREEN() << "==== DONE GMMFit::fit_1d====\n" << TermColor::RESET();
}