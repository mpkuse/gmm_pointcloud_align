#include "SlicClustering.h"


/*
 * Constructor. Nothing is done here.
 */
SlicClustering::SlicClustering() {

}

/*
 * Destructor. Clear any present data.
 */
SlicClustering::~SlicClustering() {
    clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void SlicClustering::clear_data() {
    clusters.clear();
    distances.clear();
    centers.clear();
    center_counts.clear();
    center_counts_depth.clear();
}


// #define __SlicClustering__init_data__(msg ) msg;
#define __SlicClustering__init_data__(msg ) ;
void SlicClustering::init_data( const cv::Mat& image, const cv::Mat& depth )
{
    /* asserts  */
    assert( image.rows > 0 && image.cols > 0 );
    assert( image.rows == depth.rows && image.cols == depth.cols );
    #if COLOR_CHANNEL == 1
    assert( image.channels() == 1 );
    assert( image.type() == CV_8UC1 );
    assert( depth.type() == CV_16UC1 || depth.type() == CV_32FC1 );
    #else
    assert( image.channels() == 3 );
    assert( image.type() == CV_8UC3 );
    assert( depth.type() == CV_16UC1 ||  depth.type() == CV_32FC1);
    #endif

    /* Initialize the cluster and distance matrices (ie. at pixel-wise cluster marker). */
    for (int j = 0; j < image.rows; j++) {
        vector<int> cr; cr.clear();
        vector<double> dr; dr.clear();
        for (int i = 0; i < image.cols; i++) {
            cr.push_back(-1);
            dr.push_back(FLT_MAX);
        }
        clusters.push_back(cr);
        distances.push_back(dr);
    }

    /* Initialize the centers and counters. */
    int cc = 0;
    __SlicClustering__init_data__(
    cout << "for( int j= " << step << " ; j < " << image.rows - step/2 << " ; j+= " << step << " )\n";)
    for (int j = step; j < image.rows - step/2; j += step) {
        __SlicClustering__init_data__(
        cout << "\t@j=" << j << "\tfor( int i= " << step << " ; j < " << image.cols - step/2 << " ; j+= " << step << " )\n";)
        for (int i = step; i < image.cols - step/2; i += step) {
            PixElement center; //fill this up

            __SlicClustering__init_data__(
            cout << TermColor::YELLOW() << "---\n\t\ti=" << i << "  j=" << j << TermColor::RESET() << endl;)


            // look at the neighbourhood of pixel (j,i)
            #if COLOR_CHANNEL == 3
            float sum_red = 0.0, sum_green = 0.0, sum_blue = 0.0; int sum_intensity_num = 0;
            #else
            float sum_intensity=0.0; int sum_intensity_num=0;
            #endif

            float sum_depth=0.0; int sum_depth_num=0;

            __SlicClustering__init_data__(
            cout << "\t\tloop on neighbourhood [" << j-step/2 << ", " << j+step/2 << ") X [" << i-step/2 << ", " << i+step/2 << ")\n"; )
            for( int del_j = -step/2 ; del_j < step/2 ; del_j++ ) {
                for( int del_i = -step/2 ; del_i < step/2 ; del_i++ ) {


                    assert( j+del_j >=0 && j+del_j < image.rows  &&  i+del_i >= 0 && i+del_i < image.cols );

                    #if COLOR_CHANNEL == 3
                    cv::Scalar colour = image.at<cv::Vec3b>( j+del_j, i+del_i );
                    sum_red   += (float)colour[0];
                    sum_green += (float)colour[1];
                    sum_blue  += (float)colour[2];
                    #else
                    uchar colour = image.at<uchar>(  j+del_j, i+del_i  );
                    sum_intensity += (float) colour;
                    #endif
                    sum_intensity_num++;



                    float depth_val;
                    if( depth.type() == CV_16UC1 )
                        depth_val = (float) depth.at<uint16_t>(  j+del_j, i+del_i  );
                    else if( depth.type() == CV_32FC1 )
                        depth_val = (float) depth.at<float>(  j+del_j, i+del_i  );
                    else {
                        cout << "[init_data]depth type is neighter of CV_16UC1 or CV_32FC1\n";
                        assert( false );
                        exit(1);
                    }

                    if( depth_val > 1e-3 && depth_val < 20. ) {
                        sum_depth+= depth_val;
                        sum_depth_num++;
                    }

                }
            }
            __SlicClustering__init_data__(
            cout << "\t\tneighbourhood loop done; sum_intensity_num=" << sum_intensity_num << "\tsum_depth_num=" << sum_depth_num << "\n"; )


            // fillup spatial positions
            center.u = j; //rowIdx
            center.v = i; //colIdx
            __SlicClustering__init_data__(
            cout << TermColor::GREEN() << "\t\tcenter uv := " << center.u << ", " << center.v << TermColor::RESET() << endl; )

            // fillup color/intensity
            #if COLOR_CHANNEL == 3
            center.red = sum_red / sum_intensity_num;
            center.green = sum_green / sum_intensity_num;
            center.blue = sum_blue / sum_intensity_num;
            __SlicClustering__init_data__(
            cout << TermColor::GREEN() << "\t\tcenter rgb := " << center.red  << ", " << center.green << ", " << center.blue << TermColor::RESET() << endl;)
            #else
            center.intensity = sum_intensity / sum_intensity_num;
            __SlicClustering__init_data__(
            cout << TermColor::GREEN() << "\t\tcenter intensity := " << center.intensity << TermColor::RESET() << endl; )
            #endif


            //fillup depth
            if( sum_depth_num > 0 ) {
                center.D = sum_depth / sum_depth_num;
                __SlicClustering__init_data__(
                cout << TermColor::GREEN() << "\t\tcenter depth(D) := " << center.D << TermColor::RESET() << endl;)
            } else {
                center.D = -1;
                __SlicClustering__init_data__(
                cout << TermColor::RED() << "\t\tcenter depth(D) := NAN (-1)" << center.D << TermColor::RESET() << endl; )
            }


            #if 0 // OLD CODE, eventually TODO Removal
            cv::Point nc;// = find_local_minimum(image, cvPoint(i,j)); //TODO
            nc = cv::Point( i, j ); //quick fix, just set it as center and get by

            /* Generate the center vector. */
            center.u = j; //rowIdx
            center.v = i; //colIdx

            #if 0
            // old code
            float depth_val = depth.at<uint16_t>( nc.y, nc.x ); //TODO: if Z is zero (aka invalid depth initialize cluster center with someother point. )
            #else
            float depth_val;
            if( depth.type() == CV_16UC1 )
                depth_val = depth.at<uint16_t>( nc.y, nc.x );
            else if( depth.type() == CV_32FC1 )
                depth_val = depth.at<float>( nc.y, nc.x );
            else {
                assert( false );
                cout << "depth type is neighter of CV_16UC1 or CV_32FC1\n";
                exit(1);
            }
            #endif

            if( depth_val < 1e-5 ) {
                cout << "skip, because this pixel has invalid depth\n";
                continue;
            }
            // back_project( center.v, center.u,  depth_val,  center.X, center.Y, center.Z );
            center.D = depth_val;

            #if COLOR_CHANNEL == 3
            cv::Scalar colour = image.at<cv::Vec3b>( nc.y, nc.x );
            center.red = colour[0];
            center.green = colour[1];
            center.blue = colour[2];
            #else
            uchar colour = image.at<uchar>( nc.y, nc.x );
            center.intensity = (float) colour;
            #endif
            #endif // OLD_CODE


            /* Append to vector of centers. */
            __SlicClustering__init_data__(
            cout << "init cluster#"  <<  cc << " :\t";
            PixElement::pretty_print( center );
            )
            centers.push_back(center);
            center_counts.push_back(0);
            center_counts_depth.push_back(0);
            cc++;
        }
    }
}


// #define __SlicClustering__generate_superpixels( msg ) msg;
#define __SlicClustering__generate_superpixels( msg ) ;
void SlicClustering::generate_superpixels(
    const cv::Mat& image, const cv::Mat& depth,
    int step
     )
{
    this->step = step;

    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image, depth);

    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < NR_ITERATIONS; i++) {
        __SlicClustering__generate_superpixels(
        cout << ">>>>>>>>>>>>>>>>> EM Iteration#" << i << endl;
        cout << "\t--Reset distance values.\n";)
        /* Reset distance values. */
        for (int k = 0;k < image.rows; k++) {
            for (int j = 0; j < image.cols; j++) {
                distances[k][j] = FLT_MAX;
            }
        }

        /* Only compare to pixels in a 2 x step by 2 x step region. */
        __SlicClustering__generate_superpixels(
        cout << "\t--Only compare to pixels in a 2 x step by 2 x step region.\n"; )
        for (int j = 0; j < (int) centers.size(); j++) {
            // cout << "@j (centersIdx)=" << j << "\t";
            // cout << "k(rowsIdx)= " << centers[j].u - step  << "  --->  " << centers[j].u + step << "\t";
            // cout << "l(colIdx)= " << centers[j].v - step  << "  --->  " << centers[j].v + step << "\t";
            // cout << endl;

            for (int k = centers[j].u - step; k < centers[j].u + step; k++) {
                for (int l = centers[j].v - step; l < centers[j].v + step; l++) {

                    if (k >= 0 && k < image.rows && l >= 0 && l < image.cols ) {
                        PixElement tmp;
                        tmp.u = k; tmp.v = l;


                        float depth_val;
                        if( depth.type() == CV_16UC1 )
                            depth_val = depth.at<uint16_t>( k, l );
                        else if( depth.type() == CV_32FC1 )
                            depth_val = depth.at<float>( k, l );
                        else {
                            assert( false );
                            cout << "depth type is neighter of CV_16UC1 or CV_32FC1\n";
                            exit(1);
                        }

                        // back_project( l, k,  depth_val,  tmp.X, tmp.Y, tmp.Z );
                        if( depth_val < 1e-3 || depth_val > 25. )
                            tmp.D = -1.0;
                        else
                            tmp.D = depth_val;


                        #if COLOR_CHANNEL == 3
                        cv::Scalar colour = image.at< cv::Vec3b >( k, l );
                        tmp.red = colour[0]; tmp.green = colour[1]; tmp.blue = colour[2];
                        #else
                        tmp.intensity = (float) image.at< uchar >( k, l );
                        #endif

                        // double d = compute_dist(j, cvPoint(k,l), colour);
                        double d = PixElement::distance( centers[j], tmp );

                        /* Update cluster allocation if the cluster minimizes the
                           distance. */
                        if (d < distances[k][l]) {
                            distances[k][l] = d;
                            clusters[k][l] = j;
                        }
                    }
                }
            }
        }

        /* Clear the center values. */
        __SlicClustering__generate_superpixels(
        cout << "\t--Clear the center values.\n"; )
        for (int j = 0; j < (int) centers.size(); j++) {
            // centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
            centers[j].reset();
            center_counts[j] = 0;
            center_counts_depth[j] = 0;
        }

        /* Compute the new cluster centers. */
        __SlicClustering__generate_superpixels(
        cout << "\t--Compute the new cluster centers.\n";)
        for (int k = 0; k < image.rows; k++) {
            for (int j = 0; j < image.cols; j++) {
                int c_id = clusters[k][j];

                if (c_id != -1) {
                    PixElement tmp;
                    tmp.u = k; tmp.v = j;

                    // Add up pixel 2d locations
                    centers[c_id].u += tmp.u;
                    centers[c_id].v += tmp.v;

                    // Add up colors
                    #if COLOR_CHANNEL == 3
                    cv::Scalar colour = image.at< cv::Vec3b >( tmp.u, tmp.v );
                    tmp.red = colour[0]; tmp.green = colour[1]; tmp.blue = colour[2];
                    centers[c_id].red += tmp.red;
                    centers[c_id].green += tmp.green;
                    centers[c_id].blue += tmp.blue;
                    #else
                    tmp.intensity = (float) image.at< uchar >( tmp.u, tmp.v  );
                    centers[c_id].intensity += tmp.intensity;
                    #endif


                    // keep track of how many pixels added (to be divided later to get the mean)
                    center_counts[c_id] += 1;


                    // Handling for depth TODO, need to do this separately
                    float depth_val;
                    if( depth.type() == CV_16UC1 )
                        depth_val = (float) depth.at<uint16_t>( tmp.u, tmp.v );
                    else if( depth.type() == CV_32FC1 )
                        depth_val = (float) depth.at<float>( tmp.u, tmp.v );
                    else {
                        cout << "depth type is neighter of CV_16UC1 or CV_32FC1\n";
                        assert( false );
                        exit(1);
                    }

                    // TODO: For now just simple averaging of valid depth values is implemented.
                    //       In the future if need be, wil implement robust mean estimation for depth values.
                    if( depth_val < 1e-3 || depth_val > 25. ) {
                        //igore
                        // cout << "ignore depth value\n";
                    }
                    else {
                        centers[c_id].D += depth_val;
                        center_counts_depth[c_id] += 1;
                    }




                }
            }
        }

        #if 0
        cout << "Print the info after reassignment of centers\n";
        for (int j = 0; j < (int) centers.size(); j++) {
            cout << "cluster#" << j << " of total clusters = " << centers.size() << "\t";
            if(  center_counts[j] ==  center_counts_depth[j] )
                cout << TermColor::GREEN();
            else cout << TermColor::RED();

            cout << "centers_counts=" << center_counts[j] << "\t";
            cout << "center_counts_depth=" << center_counts_depth[j] << "\t";
            cout << endl;

            cout << TermColor::RESET();

        }
        // cout << "EXIT...\n";
        // exit(1);
        #endif


        /* Remove Clusters with no or less support */
        vector<int> idx_to_remove;
        for (int j = 0; j < (int) centers.size(); j++) {
            if( center_counts[j] < 30  || center_counts_depth[j] < 30 ) {
                // cout << "cluster#" << j << " of total clusters = " << centers.size() << " has center_counts=" <<  center_counts[j]  << " (less than 10), and center_counts_depth= " << center_counts_depth[j] << " this will cause sigsegv, so mark as Toremove\n";
                idx_to_remove.push_back( j );
            }
        }
        for( int jjj=0; jjj< (int) idx_to_remove.size() ; jjj++ ) {
            center_counts.erase( center_counts.begin() + (idx_to_remove[jjj] - jjj) );
            center_counts_depth.erase( center_counts_depth.begin() + (idx_to_remove[jjj] - jjj) );
            centers.erase( centers.begin() + (idx_to_remove[jjj] - jjj) );
            // cout << "Erase at idx:: "<< idx_to_remove[jjj] - jjj << endl;
        }


        /* Normalize the clusters. Only uv, intensity */
        __SlicClustering__generate_superpixels(
        cout << "\t--Normalize the clusters\n";)
        assert( centers.size() == center_counts.size() && centers.size() > 0 );
        for (int j = 0; j < (int) centers.size(); j++) {
            assert( center_counts[j] > 0 && (string("number of items in cluster#")+to_string(j)+" were zero.").c_str() );
            assert( center_counts_depth[j] > 0);

            centers[j].u /= center_counts[j];
            centers[j].v /= center_counts[j];
            #if COLOR_CHANNEL == 3
            centers[j].red /= center_counts[j];
            centers[j].green /= center_counts[j];
            centers[j].blue /= center_counts[j];
            #else
            centers[j].intensity /= center_counts[j];
            #endif

            centers[j].D /= center_counts_depth[j];

        }


    }
    m_generate_superpixels = true;

}




//--------------------------------------------------------------------------//
//-------------------- Visualization Functions -----------------------------//
//--------------------------------------------------------------------------//

void SlicClustering::display_center_grid(cv::Mat& image, cv::Scalar colour) {
    for (int i = 0; i < (int) centers.size(); i++) {
        // cvCircle(image, cvPoint(centers[i][3], centers[i][4]), 2, colour, 2);
        // cout << "center#" << i << "\t count = " << center_counts[i] << "\t";
        // PixElement::pretty_print( centers[i] );
        // cout << endl;

        cv::Point pt( centers[i].v, centers[i].u );
        cv::circle(image, pt, 2, colour, -1, CV_AA);

    }
}



void SlicClustering::display_center_grid() {
    cout << "[SlicClustering::display_center_grid]\n";
    for (int i = 0; i < (int) centers.size(); i++) {

        cout << "center#" << i << "\t count = " << center_counts[i] << "\t";
        PixElement::pretty_print( centers[i] );
        // cout << endl;

    }
}




void SlicClustering::colour_with_cluster_means(cv::Mat& image ) {
    assert( image.channels() == 3 );

    vector<cv::Scalar> colours(centers.size());

    /* Gather the colour values per cluster. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int index = clusters[i][j];
            cv::Scalar colour = image.at<cv::Vec3b>( i, j );

            colours[index][0] += colour[0];
            colours[index][1] += colour[1];
            colours[index][2] += colour[2];
        }
    }

    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < (int)colours.size(); i++) {
        colours[i][0] /= center_counts[i];
        colours[i][1] /= center_counts[i];
        colours[i][2] /= center_counts[i];
    }

    /* Fill in. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Scalar ncolour = colours[ clusters[i][j] ];

            cv::Vec3b tmp = cv::Vec3b( (uchar)ncolour[2], (uchar)ncolour[1], (uchar)ncolour[0] );
            image.at<cv::Vec3b>(i,j) = tmp;
        }
    }
}




void SlicClustering::display_contours( const cv::Mat& image, cv::Scalar colour, cv::Mat& output ) {
    // assert( image.channels() == 3 );
    assert( image.rows > 0 && image.cols > 0 && (image.channels()==1 || image.channels()==3));


    if( image.channels() == 3 )
        output = image.clone();
    else
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);



    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	/* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
	vector<cv::Point> contours;
	vec2db istaken;
	for (int i = 0; i < image.rows; i++) {
        vector<bool> nb;
        for (int j = 0; j < image.cols; j++) {
            nb.push_back(false);
        }
        istaken.push_back(nb);
    }

    /* Go through all the pixels. */
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int nr_p = 0;

            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];

                if (x >= 0 && x < image.rows && y >= 0 && y < image.cols) {
                    if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                        nr_p += 1;
                    }
                }
            }

            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(j,i));
                istaken[i][j] = true;
            }
        }
    }

    /* Draw the contour pixels. */
    cv::Vec3b colo = cv::Vec3b( colour[0], colour[1], colour[2] );
    for (int i = 0; i < (int)contours.size(); i++) {
        // cvSet2D(image, contours[i].y, contours[i].x, colour);
        output.at< cv::Vec3b >( contours[i] ) = colo;
    }
}




//--------------------------------------------------------------------------//
//-------------------- Retrive Functions -----------------------------------//
//--------------------------------------------------------------------------//
// rowcol_or_xy: true will give row,col; false will give xy
MatrixXd SlicClustering::retrive_superpixel_uv( bool return_homogeneous, bool rowcol_or_xy ) // 2xN matrix or 3xN
{
    assert( m_generate_superpixels && "you are calling retrive function before solving" );
    MatrixXd to_ret;
    if( return_homogeneous )
        to_ret = MatrixXd::Ones( 3, centers.size() );
    else
        to_ret = MatrixXd::Ones( 2, centers.size() );

    for (int i = 0; i < (int) centers.size(); i++) {

        if( rowcol_or_xy ) {
        to_ret( 0, i ) = centers[i].u;
        to_ret( 1, i ) = centers[i].v;
        }
        else{
            to_ret( 0, i ) = centers[i].v;
            to_ret( 1, i ) = centers[i].u;
        }
    }

    return to_ret;
}



MatrixXd SlicClustering::retrive_superpixel_XYZ(  bool return_homogeneous ) // 3xN matrix, or 4xN
{
    assert( m_generate_superpixels && "you are calling retrive function before solving" );
    MatrixXd to_ret;

    if( return_homogeneous )
        to_ret = MatrixXd::Ones( 4, centers.size() );
    else
        to_ret = MatrixXd::Ones( 3, centers.size() );

    for (int i = 0; i < (int) centers.size(); i++) {
        float __X, __Y, __Z;
        back_project( centers[i].v, centers[i].u, centers[i].D,  __X, __Y, __Z );
        to_ret( 0, i ) = __X;
        to_ret( 1, i ) = __Y;
        to_ret( 2, i ) = __Z;
    }

    return to_ret;
}


int SlicClustering::retrive_nclusters()
{
    return (int) centers.size();
}

MatrixXd SlicClustering::retrive_superpixel_localnormals()
{
    // for a given superpixel i, the surface normal at this point :
    // \sum_{pixel u_j is in superpixel i} ( 3dpt_of_superpixelcenter -  3dpts_of_u_j )
    //           ^^^^ Need to repeat this process for each super pixel
    cout << "NOT IMPLEMENTED...\n";
    exit(1);
}
