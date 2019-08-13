#pragma once



#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <math.h>
#include <vector>

#include <iostream>
#include <iomanip>
using namespace std;

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>


/* 2d matrices are handled by 2d vectors. */
#define vec2dd vector<vector<double> >
#define vec2di vector<vector<int> >
#define vec2db vector<vector<bool> >

/* The number of iterations run by the clustering algorithm. */
#define NR_ITERATIONS 3


// can be either 1 or 3
#define COLOR_CHANNEL 1
// #define COLOR_CHANNEL 3

/* A Pixel is represented by this.. */
class PixElement {
public:
    int u, v; //< spatial position
    float X, Y, Z; //< 3d position

    #if COLOR_CHANNEL == 1
    float intensity;
    #else
    float red, green, blue;
    #endif


    void reset() {
        u = 0; v=0; X=Y=Z=0 ;
        #if COLOR_CHANNEL == 1
        intensity = 0;
        #else
        red=green=blue=0;
        #endif
    }


    static float distance( const PixElement& a, const PixElement& b )
    {
        float d_spatial = (a.u - b.u)*(a.u - b.u) + (a.v - b.v)*(a.v - b.v);
        float d_volume  = (a.X - b.X)*(a.X - b.X) + (a.Y - b.Y)*(a.Y - b.Y) + (a.Z - b.Z)*(a.Z - b.Z);

        #if COLOR_CHANNEL == 1
        float d_color   = (a.intensity - b.intensity) * (a.intensity - b.intensity);
        #else
        float d_color = (a.red - b.red)*(a.red - b.red) + (a.green - b.green)*(a.green - b.green) + (a.blue - b.blue)*(a.blue - b.blue) ;
        #endif

        // squared weights
        float w_spatial = pow( (  1.0 / 100. ), 2 ) ;
        float w_volume  = pow( (  0.5 / 10000. ), 2 ) ;
        float w_color   = pow( (  1.0 / 100. ), 2 ) ;

        float d = w_spatial*d_spatial + w_volume*d_volume + w_color*d_color;
        return d;
        return sqrt(d); //TODO:try returning squared distance. if you just want to compare no point taking the square root
    }

    static void pretty_print( const PixElement& a )
    {
        cout << "u=" << a.u << ", v=" << a.v << "\t";
        cout << "X=" << a.X << ", Y=" << a.Y << ", Z=" << a.Z << "\t";
        #if COLOR_CHANNEL == 1
        cout << "intensity=" << a.intensity << endl;
        #else
        cout << "red=" << a.red << ", green=" << a.green << ", blue=" << a.blue << endl;
        #endif
    }

};


class SlicClustering {
public:
    /* Class constructors and deconstructors. */
    SlicClustering();
    ~SlicClustering();
    void clear_data();

    void init_data( const cv::Mat& image, const cv::Mat& depth );

    //image, depth image (cv_16uc1) the stepsize (int), and the weight (int).
    void generate_superpixels(   const cv::Mat& image, const cv::Mat& depth, int step, int nc );


    // viz
    void display_center_grid();
    void display_center_grid(cv::Mat& image, cv::Scalar colour);
    void colour_with_cluster_means(cv::Mat& image);
    void display_contours(cv::Mat& image, cv::Scalar colour);

    // data extractor
    MatrixXd retrive_superpixel_uv( bool return_homogeneous = false ); // 2xN matrix or 3xN
    MatrixXd retrive_superpixel_XYZ(  bool return_homogeneous = false  ); // 3xN matrix, or 4xN

private:
    /* The cluster assignments and distance values for each pixel. */
    vec2di clusters;
    vec2dd distances;

    /* The LAB and xy values of the centers. */
    // vec2dd centers;
    vector<PixElement> centers;
    /* The number of occurences of each center. */
    vector<int> center_counts;

    /* The step size per cluster, and the colour (nc) and distance (ns)
     * parameters. */
    int step, nc, ns;


    // TODO : Write functions to set these values
    float fx = 385.7544860839844f;
    float fy = 385.7544860839844f;
    float cx = 323.1204833984375f;
    float cy = 236.7432098388672f;


    void back_project(
    const float &u, const float &v, const float &depth, float &x, float &y, float &z)
    {
        x = (u - cx) / fx * depth;
        y = (v - cy) / fy * depth;
        z = depth;
    }

    bool m_generate_superpixels = false;

};
