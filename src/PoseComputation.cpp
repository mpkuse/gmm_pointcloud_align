#include "PoseComputation.h"

#define _PoseComputation__closedFormSVD_(msg) msg;
// #define _PoseComputation__closedFormSVD_(msg) ;

// #define _PoseComputation__closedFormSVD_debug(msg) msg;
#define _PoseComputation__closedFormSVD_debug(msg) ;
bool PoseComputation::closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b )
{
    assert( aX.rows() == 4 && bX.rows() == 4 && aX.cols() == bX.cols() && aX.cols() > 0 );
    _PoseComputation__closedFormSVD_( ElapsedTime _t; _t.tic();
    cout << TermColor::iGREEN() << "=== PoseComputation::closedFormSVD input size = " << aX.rows() << "x" << aX.cols() << TermColor::RESET() << endl;
    )

    // centroids
    VectorXd cen_aX = aX.rowwise().mean();
    VectorXd cen_bX = bX.rowwise().mean();
    _PoseComputation__closedFormSVD_debug(
    cout << "centroids computed: ";
    cout << "cen_aX = " << cen_aX.rows() << "x" << cen_aX.cols() << "\t" << cen_aX.transpose() << "\n";
    cout << "cen_bX = " << cen_bX.rows() << "x" << cen_bX.cols() << "\t" << cen_bX.transpose() << "\n";
    )


    // aX_cap:= aX - cen_aX
    // bX_cap:= bX - cen_bX
    // H := aX_cap * bX_cap.transpose()
    Matrix3d H = (bX.colwise() - cen_bX).topRows(3) * (aX.colwise() - cen_aX).topRows(3).transpose();
    _PoseComputation__closedFormSVD_debug(cout << "H=" << H.rows() << "x" << H.cols() << endl;)

    // U,S,Vt = svd( H )
    JacobiSVD<Matrix3d> svd( H, ComputeFullU | ComputeFullV);

    _PoseComputation__closedFormSVD_debug(
    cout << "Singular=\n" << svd.singularValues() << endl;
    cout << "U=\n" << svd.matrixU()  << endl;
    cout << "V=\n" << svd.matrixV()  << endl;
    )

    // R := V * Ut. if det(R) is -1, then use R = [v1, v2, -v3] * Ut
    Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    _PoseComputation__closedFormSVD_debug(
    cout << "R=\n" << R << endl;
    cout << "R.det=" << R.determinant() << endl;)

    // assert( abs(R.determinant()-1.0)<1e-6 );
    if( abs(R.determinant()+1.0)<1e-6 ) // then determinant is -1
    {
        Matrix3d _V = svd.matrixV();
        for( int fd=0;fd<3;fd++)
            _V(fd,2) = -_V(fd,2);
        _PoseComputation__closedFormSVD_debug( cout << "_V=\n" << _V; )
        R = _V * svd.matrixU().transpose();
    }

    // translation : mean(aX) - R mean(bX)
    Vector3d tr = aX.rowwise().mean().topRows(3) - R * bX.rowwise().mean().topRows(3);
    _PoseComputation__closedFormSVD_debug(cout << "Translation=" << tr << endl;)


    a_T_b = Matrix4d::Identity();
    a_T_b.topLeftCorner(3,3) = R;
    a_T_b.col(3).topRows(3) = tr;

    _PoseComputation__closedFormSVD_(
    cout << "a_T_b=\n" << a_T_b << endl;
    cout << TermColor::BLUE() << "[PoseComputation::closedFormSVD]computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl; )
    return true;
}

#define _PoseComputation__refine_info( msg ) msg;
// #define _PoseComputation__refine_info( msg ) ;

// #define _PoseComputation__refine_debug( msg ) msg;
#define _PoseComputation__refine_debug( msg ) ;
bool PoseComputation::refine( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b )
{
    _PoseComputation__refine_info(
    cout << TermColor::iGREEN() << "=== PoseComputation::refine input size = " << aX.rows() << "x" << aX.cols() << TermColor::RESET() << endl;
    ElapsedTime _t; _t.tic();
    )

    //--- Initial Guess
    Matrix4d a_Tcap_b = a_T_b;
    double T_cap_q[10], T_cap_t[10]; //quaternion and translation
    PoseManipUtils::eigenmat_to_raw( a_Tcap_b, T_cap_q, T_cap_t );
    _PoseComputation__refine_info(
        cout << "Initial Estimate: " << PoseManipUtils::prettyprintMatrix4d( a_Tcap_b ) << endl;)

    // switch constraints
    double * s = new double [aX.cols()];
    for( int i=0; i<aX.cols() ; i++ ) s[i] = 1.0;

    //--- Setup Residues
    ceres::Problem problem;
    for( int i=0 ; i<aX.cols() ; i++ )
    {
        auto norm = new CauchyLoss(.1); // NULL
        // CostFunction* cost_function = EuclideanDistanceResidue::Create( aX.col(i).head(3), bX.col(i).head(3) );
        // problem.AddResidualBlock( cost_function, NULL, T_cap_q, T_cap_t );


        CostFunction* cost_function = EuclideanDistanceResidueSwitchingConstraint::Create( aX.col(i).head(3), bX.col(i).head(3) );
        problem.AddResidualBlock( cost_function, norm, T_cap_q, T_cap_t, &s[i] );
    }
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( T_cap_q, quaternion_parameterization );



    //--- Solve
    Solver::Options options;
    // TODO set dense solver as this is a small problem
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::DENSE_QR;
    _PoseComputation__refine_debug( options.minimizer_progress_to_stdout = true; )
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    _PoseComputation__refine_debug( std::cout << summary.FullReport() << "\n"; );
    _PoseComputation__refine_info( std::cout << summary.BriefReport() << endl; );

    //--- Retrive Solution
    PoseManipUtils::raw_to_eigenmat( T_cap_q, T_cap_t, a_T_b );
    _PoseComputation__refine_info(
        cout << "Final Pose Estimate (a_T_b): " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;)


    int n_quantiles = 4;
    int * quantile = new int[n_quantiles];
    for( int i=0 ; i<n_quantiles; i++ ) quantile[i] = 0;
    for( int i=0; i<aX.cols() ; i++ ) {

        quantile[ (int) ( (s[i]-0.001) * n_quantiles)  ]++;

        _PoseComputation__refine_debug(
        cout << std::setw(4) <<  i << ":" << std::setw(4) << std::setprecision(2) << s[i] << "\t";
        if( i%10==0) cout << endl;)
    }
    _PoseComputation__refine_debug(     cout << std::setprecision(18) << endl; )

    _PoseComputation__refine_info(
    cout << "Quantiles range=[0,4], n_quantiles=" << n_quantiles<< ", total_points=" << aX.cols() << ":\n";
    for( int i=0 ; i<n_quantiles; i++ ) cout << "\tquantile[" << i << "] = " << std::setw(5) << quantile[i] << "\tfrac=" << std::setprecision(2) <<  float(quantile[i])/aX.cols() << endl;)

    _PoseComputation__refine_info(
    cout << TermColor::BLUE() << "computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl;
    cout << TermColor::iGREEN() << "=== PoseComputation::refine Finished" << TermColor::RESET() << endl;
    )


    delete [] quantile;
    delete [] s;
    return true;
}




//----------------------------------------------------------
#define _PoseComputation__closedFormSVD_2_info( msg ) msg;
// #define _PoseComputation__closedFormSVD_2_info( msg ) ;

// #define _PoseComputation__closedFormSVD_2_debug( msg ) msg;
#define _PoseComputation__closedFormSVD_2_debug( msg ) ;

bool PoseComputation::closedFormSVD( const MatrixXd& aX_sans_depth, const VectorXd& d_a,
                           const MatrixXd& bX_sans_depth, const VectorXd& d_b,
                           const VectorXd& sf,
                           Matrix4d& a_T_b
                       )
{
    _PoseComputation__closedFormSVD_2_info(
    ElapsedTime _t; _t.tic();
    cout << TermColor::iGREEN() << "[PoseComputation::closedFormSVD] with weights, ";
    cout << "N=" << d_a.size() << "\tsf.sum()=" << sf.sum() << endl;
    cout << TermColor::RESET() << endl;
    )


    int N = d_a.size();
    assert( N > 0 );
    assert( aX_sans_depth.cols() == N && bX_sans_depth.cols() == N && d_b.size() == N && sf.size() == N );
    assert( aX_sans_depth.rows() == 4 && bX_sans_depth.rows() == 4 );
    // assert( abs(aX_sans_depth(0,3) * d_a(0) - 1.0 ) < 1e-6 );
    // assert( abs(bX_sans_depth(0,3) * d_b(0) - 1.0 ) < 1e-6 );

    _PoseComputation__closedFormSVD_2_debug(
    cout << TermColor::iWHITE() << "---" << TermColor::RESET() << endl;
    cout << "N=" << N << " but only showing first 10\n";
    cout << "aX_sans_depth:\n" << aX_sans_depth.leftCols(10) << endl;
    cout << "bX_sans_depth:\n" << bX_sans_depth.leftCols(10) << endl;
    cout << TermColor::CYAN() ;
    cout << "d_a:\t" << d_a.topRows(10).transpose() << endl;
    cout << "d_b:\t" << d_b.topRows(10).transpose() << endl;
    cout << "sf:\t" << sf.topRows(10).transpose() << endl;
    cout << "---\n" << TermColor::RESET();
    )




    //--- make aX, bX using `aX_sans_depth * d_a` and `bX_sans_depth * d_b`
    MatrixXd aX = MatrixXd::Zero( 4, N );
    MatrixXd bX = MatrixXd::Zero( 4, N );
    for( int r=0 ; r<4; r++ )
        aX.row(r) = aX_sans_depth.row(r).cwiseProduct( d_a.transpose() );

    for( int r=0 ; r<4; r++ )
        bX.row(r) = bX_sans_depth.row(r).cwiseProduct( d_b.transpose() );

    _PoseComputation__closedFormSVD_2_debug(
    cout << "aX_" << aX.rows() << "x" << aX.cols() << ":\n" << aX.leftCols(10) << endl;
    cout << "bX_" << bX.rows() << "x" << bX.cols() << ":\n" << bX.leftCols(10) << endl;
    )


    //--- centroids (weighted avg)
    Vector4d cen_aX = Vector4d::Zero();
    Vector4d cen_bX = Vector4d::Zero();

    for( int r=0 ; r<4 ; r++ )
        cen_aX.row(r) = (aX.row(r) * sf) / sf.sum();

    for( int r=0 ; r<4 ; r++ )
        cen_bX.row(r) = (bX.row(r) * sf )/ sf.sum();

    _PoseComputation__closedFormSVD_2_debug(
    cout << "centroids computed:\n";
    cout << "cen_aX = " << cen_aX.rows() << "x" << cen_aX.cols() << "\t" << cen_aX.transpose() << "\n";
    cout << "cen_bX = " << cen_bX.rows() << "x" << cen_bX.cols() << "\t" << cen_bX.transpose() << "\n";
    )



    // aX_cap:= aX - cen_aX
    // bX_cap:= bX - cen_bX
    // H := aX_cap * bX_cap.transpose()
    Matrix3d H = (bX.colwise() - cen_bX).topRows(3) * sf.asDiagonal() * (aX.colwise() - cen_aX).topRows(3).transpose();
    _PoseComputation__closedFormSVD_2_debug( cout << "H:\n" << H << endl; )


    // U,S,Vt = svd( H )
    JacobiSVD<Matrix3d> svd( H, ComputeFullU | ComputeFullV);

    _PoseComputation__closedFormSVD_2_debug(
    cout << "Singular=\n" << svd.singularValues() << endl;
    cout << "U=\n" << svd.matrixU()  << endl;
    cout << "V=\n" << svd.matrixV()  << endl;
    )

    // R := V * Ut. if det(R) is -1, then use R = [v1, v2, -v3] * Ut
    Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    _PoseComputation__closedFormSVD_2_debug(
    cout << "R=\n" << R << endl;
    cout << "R.det=" << R.determinant() << endl;)

    // assert( abs(R.determinant()-1.0)<1e-6 );
    if( abs(R.determinant()+1.0)<1e-6 ) // then determinant is -1
    {
        Matrix3d _V = svd.matrixV();
        for( int fd=0;fd<3;fd++)
            _V(fd,2) = -_V(fd,2);
        _PoseComputation__closedFormSVD_2_debug( cout << "_V=\n" << _V; )
        R = _V * svd.matrixU().transpose();
    }

    // translation : mean(aX) - R mean(bX)
    Vector3d tr = Vector3d::Zero();
    // tr = aX.rowwise().mean().topRows(3) - R * bX.rowwise().mean().topRows(3);
    MatrixXd bX_rot = R * bX.topRows(3);
    for( int g=0;g<3;g++ ) { //weighted sum
        double t1 = (aX.row(g) * sf) ;
        double t2 = (bX_rot.row(g) * sf);
        tr(g) = t1 / sf.sum() -  t2 / sf.sum();
    }
    _PoseComputation__closedFormSVD_2_debug( cout << "Translation=" << tr.transpose() << endl; )


    a_T_b = Matrix4d::Identity();
    a_T_b.topLeftCorner(3,3) = R;
    a_T_b.col(3).topRows(3) = tr;

    _PoseComputation__closedFormSVD_2_info(
    cout << "a_T_b=\n" << a_T_b << endl;
    cout << TermColor::BLUE() << "[PoseComputation::closedFormSVD:_2]computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl;
    )
    return true;
}

bool PoseComputation::refine( const MatrixXd& aX_sans_depth, VectorXd& d_a,
                   const MatrixXd& bX_sans_depth, VectorXd& d_b,
                   VectorXd& sf,
                   Matrix4d& a_T_b,
                   bool refine_pose, bool refine_depth, bool refine_switch_weights
      )
{
    int N = aX_sans_depth.cols();
    assert( d_a.size() == N && bX_sans_depth.cols() == N && d_b.size() == N && sf.size() == N );

    cout << TermColor::iYELLOW() << "==== PoseComputation::refine 2 ; N=" << N << "===\n" << TermColor::RESET();



    // --- initial guess
    double * dd_a = new double[N];
    double * dd_b = new double[N];
    double * dsf  = new double[N];

    for( int i=0 ; i<N; i++ ) {
        dd_a[i] = d_a(i);
        dd_b[i] = d_b(i);
        dsf[i]  = sf(i);
    }


    Matrix4d a_Tcap_b = a_T_b;
    double T_cap_q[10], T_cap_t[10]; //quaternion and translation
    PoseManipUtils::eigenmat_to_raw( a_Tcap_b, T_cap_q, T_cap_t );
    cout << "Initial Estimate: " << PoseManipUtils::prettyprintMatrix4d( a_Tcap_b ) << endl;



    // --- setup residues
    ceres::Problem problem;
    #if 0
    MatrixXd aX, bX;
    aX = MatrixXd::Zero( 4, N );
    bX = MatrixXd::Zero( 4, N );
    for( int r=0 ; r<4; r++ ) {
        aX.row(r) = aX_sans_depth.row(r).cwiseProduct( d_a.transpose() );
        bX.row(r) = bX_sans_depth.row(r).cwiseProduct( d_b.transpose() );
    }
    #endif
    for( int i=0 ; i<N ; i++ ) {
        if( dsf[i] < 0.1 ) //dont add if weight is too less
            continue;
        CostFunction* cost_function =
        EuclideanDistanceResidueSwitchingConstraintAndDepthRefinement::Create(
            aX_sans_depth.col(i).head(3), bX_sans_depth.col(i).head(3)
        );
        problem.AddResidualBlock( cost_function, NULL, T_cap_q, T_cap_t, &dsf[i], &dd_a[i], &dd_b[i] );
    }
    ceres::LocalParameterization *quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( T_cap_q, quaternion_parameterization );

    // --- set constant as per the flags
    cout << "\trefine_pose=" << refine_pose << "\trefine_depth=" << refine_depth << "\trefine_switch_weights=" << refine_switch_weights << endl;
    for( int i=0 ; i<N ; i++ ) {
        if( dsf[i] < 0.1 ) //dont add if weight is too less
            continue;

        if( refine_switch_weights == false )
            problem.SetParameterBlockConstant( &dsf[i] );

        if( refine_depth == false ) {
            problem.SetParameterBlockConstant( &dd_a[i] );
            problem.SetParameterBlockConstant( &dd_b[i] );
        }
    }
    if( refine_pose == false ) {
        problem.SetParameterBlockConstant( T_cap_q );
        problem.SetParameterBlockConstant( T_cap_t );
    }


    // --- solve
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
    std::cout << summary.BriefReport() << "\n";

    // --- print before and after opt
    for( int i=0; i<N ; i++ ) {
        cout << std::setw(4) <<  i << ":" << std::setw(4) << std::setprecision(3) << dsf[i] << ":" << std::setw(4)  << dd_a[i] << "," << std::setw(4) << dd_b[i] << ";";
        cout << TermColor::YELLOW() <<  "{" << std::setw(4) << std::setprecision(3) << sf(i) << ":" << std::setw(4) << d_a(i) << "," << std::setw(4) << d_b(i) << "}\t" << TermColor::RESET();
        if( i%3==0) cout << endl;
    }
    cout << std::setprecision(18) << endl;

    // --- retrive solutions
    for( int i=0 ; i<N; i++ ) {
        d_a(i) = dd_a[i];
        d_b(i) = dd_b[i];
        sf(i)  = dsf[i] ;
    }

    PoseManipUtils::raw_to_eigenmat( T_cap_q, T_cap_t, a_T_b );
    cout << "Final Pose Estimate (a_T_b): " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;


        int n_quantiles = 4;
        int * quantile = new int[n_quantiles];
        for( int i=0 ; i<n_quantiles; i++ ) quantile[i] = 0;
        for( int i=0; i<N ; i++ ) {

            quantile[ (int) ( (dsf[i]-0.001) * n_quantiles)  ]++;

        }
        cout << std::setprecision(18) << endl;


        cout << "Quantiles range=[0,4], n_quantiles=" << n_quantiles<< ", total_points=" << N << ":\n";
        for( int i=0 ; i<n_quantiles; i++ ) cout << "\tquantile[" << i << "] = " << std::setw(5) << quantile[i] << "\tfrac=" << std::setprecision(2) <<  float(quantile[i])/N << endl;


        delete [] quantile;
        delete [] dsf;
        delete [] dd_a;
        delete [] dd_b;
        cout << TermColor::iYELLOW() << "==== PoseComputation::refine 2 FINISHED ; N=" << N << "===\n" << TermColor::RESET();


}
