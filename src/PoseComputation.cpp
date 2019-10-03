#include "PoseComputation.h"

// #define _PoseComputation__closedFormSVD_(msg) msg;
#define _PoseComputation__closedFormSVD_(msg) ;
bool PoseComputation::closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b )
{
    assert( aX.rows() == 4 && bX.rows() == 4 && aX.cols() == bX.cols() && aX.cols() > 0 );
    _PoseComputation__closedFormSVD_( ElapsedTime _t; _t.tic();
    cout << TermColor::GREEN() << "=== PoseComputation::closedFormSVD input size = " << aX.rows() << "x" << aX.cols() << TermColor::RESET() << endl;
    )

    // centroids
    VectorXd cen_aX = aX.rowwise().mean();
    VectorXd cen_bX = bX.rowwise().mean();
    _PoseComputation__closedFormSVD_(
    cout << "centroids computed: ";
    cout << "cen_aX = " << cen_aX.rows() << "x" << cen_aX.cols() << "\t" << cen_aX.transpose() << "\n";
    cout << "cen_bX = " << cen_bX.rows() << "x" << cen_bX.cols() << "\t" << cen_bX.transpose() << "\n";
    )


    // aX_cap:= aX - cen_aX
    // bX_cap:= bX - cen_bX
    // H := aX_cap * bX_cap.transpose()
    Matrix3d H = (bX.colwise() - cen_bX).topRows(3) * (aX.colwise() - cen_aX).topRows(3).transpose();
    _PoseComputation__closedFormSVD_(cout << "H=" << H.rows() << "x" << H.cols() << endl;)

    // U,S,Vt = svd( H )
    JacobiSVD<Matrix3d> svd( H, ComputeFullU | ComputeFullV);

    _PoseComputation__closedFormSVD_(
    cout << "Singular=\n" << svd.singularValues() << endl;
    cout << "U=\n" << svd.matrixU()  << endl;
    cout << "V=\n" << svd.matrixV()  << endl;
    )

    // R := V * Ut. if det(R) is -1, then use R = [v1, v2, -v3] * Ut
    Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    _PoseComputation__closedFormSVD_(
    cout << "R=\n" << R << endl;
    cout << "R.det=" << R.determinant() << endl;)

    // assert( abs(R.determinant()-1.0)<1e-6 );
    if( abs(R.determinant()+1.0)<1e-6 ) // then determinant is -1
    {
        Matrix3d _V = svd.matrixV();
        for( int fd=0;fd<3;fd++)
            _V(fd,2) = -_V(fd,2);
        cout << "_V=\n" << _V;
        R = _V * svd.matrixU().transpose();
    }

    // translation : mean(aX) - R mean(bX)
    Vector3d tr = aX.rowwise().mean().topRows(3) - R * bX.rowwise().mean().topRows(3);
    _PoseComputation__closedFormSVD_(cout << "Translation=" << tr << endl;)


    a_T_b = Matrix4d::Identity();
    a_T_b.topLeftCorner(3,3) = R;
    a_T_b.col(3).topRows(3) = tr;

    _PoseComputation__closedFormSVD_(
    cout << "a_T_b=\n" << a_T_b << endl;
    cout << TermColor::BLUE() << "[PoseComputation::closedFormSVD]computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl; )
    return true;
}

#define _PoseComputation__align3D3DWithRefinement_info( msg ) msg;
// #define _PoseComputation__align3D3DWithRefinement_info( msg ) ;

// #define _PoseComputation__align3D3DWithRefinement_debug( msg ) msg;
#define _PoseComputation__align3D3DWithRefinement_debug( msg ) ;
bool PoseComputation::align3D3DWithRefinement( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b )
{
    _PoseComputation__align3D3DWithRefinement_info(
    cout << TermColor::iGREEN() << "=== PoseComputation::align3D3DWithRefinement input size = " << aX.rows() << "x" << aX.cols() << TermColor::RESET() << endl;
    ElapsedTime _t; _t.tic();
    )

    //--- Initial Guess
    Matrix4d a_Tcap_b = Matrix4d::Identity();
    closedFormSVD( aX, bX, a_Tcap_b );
    double T_cap_q[10], T_cap_t[10]; //quaternion and translation
    PoseManipUtils::eigenmat_to_raw( a_Tcap_b, T_cap_q, T_cap_t );
    _PoseComputation__align3D3DWithRefinement_info(
        cout << "Initial Estimate: " << PoseManipUtils::prettyprintMatrix4d( a_Tcap_b ) << endl;)

    // switch constraints
    double * s = new double [aX.cols()];
    for( int i=0; i<aX.cols() ; i++ ) s[i] = 1.0;

    //--- Setup Residues
    ceres::Problem problem;
    for( int i=0 ; i<aX.cols() ; i++ )
    {
        // CostFunction* cost_function = EuclideanDistanceResidue::Create( aX.col(i).head(3), bX.col(i).head(3) );
        // problem.AddResidualBlock( cost_function, NULL, T_cap_q, T_cap_t );
        // problem.AddResidualBlock( cost_function, new CauchyLoss(.01), T_cap_q, T_cap_t );

        CostFunction* cost_function = EuclideanDistanceResidueSwitchingConstraint::Create( aX.col(i).head(3), bX.col(i).head(3) );
        problem.AddResidualBlock( cost_function, NULL, T_cap_q, T_cap_t, &s[i] );
        // problem.AddResidualBlock( cost_function, new CauchyLoss(.1), T_cap_q, T_cap_t, &s[i] );
    }
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( T_cap_q, quaternion_parameterization );



    //--- Solve
    Solver::Options options;
    // TODO set dense solver as this is a small problem
    options.minimizer_progress_to_stdout = false;
    _PoseComputation__align3D3DWithRefinement_debug( options.minimizer_progress_to_stdout = true; )
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    _PoseComputation__align3D3DWithRefinement_debug( std::cout << summary.FullReport() << "\n"; );
    _PoseComputation__align3D3DWithRefinement_info( std::cout << summary.BriefReport() << endl; );

    //--- Retrive Solution
    PoseManipUtils::raw_to_eigenmat( T_cap_q, T_cap_t, a_T_b );
    _PoseComputation__align3D3DWithRefinement_info(
        cout << "Final Pose Estimate (a_T_b): " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;)


    int n_quantiles = 4;
    int * quantile = new int[n_quantiles];
    for( int i=0 ; i<n_quantiles; i++ ) quantile[i] = 0;
    for( int i=0; i<aX.cols() ; i++ ) {

        quantile[ (int) ( (s[i]-0.001) * n_quantiles)  ]++;

        _PoseComputation__align3D3DWithRefinement_debug(
        cout << std::setw(4) <<  i << ":" << std::setw(4) << std::setprecision(2) << s[i] << "\t";
        if( i%10==0) cout << endl;)
    }
    _PoseComputation__align3D3DWithRefinement_debug(     cout << std::setprecision(18) << endl; )

    _PoseComputation__align3D3DWithRefinement_info(
    cout << "Quantiles range=[0,4], n_quantiles=" << n_quantiles<< ", total_points=" << aX.cols() << ":\n";
    for( int i=0 ; i<n_quantiles; i++ ) cout << "\tquantile[" << i << "] = " << std::setw(5) << quantile[i] << "\tfrac=" << std::setprecision(2) <<  float(quantile[i])/aX.cols() << endl;)

    _PoseComputation__align3D3DWithRefinement_info(
    cout << TermColor::BLUE() << "computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl;
    cout << TermColor::iGREEN() << "=== PoseComputation::align3D3DWithRefinement Finished" << TermColor::RESET() << endl;
    )


    delete [] quantile;
    delete [] s;
    return true;
}
