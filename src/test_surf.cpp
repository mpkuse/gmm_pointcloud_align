/OPENCV C++ Tutorial:Object Detection Using SURF detector
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;

int main()
{
  //Load the Images
  Mat image_obj = imread( "C:\\Users\\arjun\\Desktop\\image_object.png", CV_LOAD_IMAGE_GRAYSCALE );
  Mat image_scene = imread( "C:\\Users\\arjun\\Desktop\\background_scene.png", CV_LOAD_IMAGE_GRAYSCALE );

  //Check whether images have been loaded
  if( !image_obj.data)
  {
   cout<< " --(!) Error reading image1 " << endl;
   return -1;
  }
   if( !image_scene.data)
  {
   cout<< " --(!) Error reading image2 " << endl;
   return -1;
  }


  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;
  SurfFeatureDetector detector( minHessian);
  vector<KeyPoint> keypoints_obj,keypoints_scene;
  detector.detect( image_obj, keypoints_obj );
  detector.detect( image_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;
  Mat descriptors_obj, descriptors_scene;
  extractor.compute( image_obj, keypoints_obj, descriptors_obj );
  extractor.compute( image_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_obj, descriptors_scene, matches );

   Mat img_matches;
  drawMatches( image_obj, keypoints_obj, image_scene, keypoints_scene,
               matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //--Step3: Show detected (drawn) keypoints
  imshow("DetectedImage", img_matches );
  waitKey(0);

  return 0;
  }
