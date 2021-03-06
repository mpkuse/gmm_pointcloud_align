cmake_minimum_required(VERSION 2.8.3)
project(gmm_pointcloud_align)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  sensor_msgs
  cv_bridge
  message_filters
  camera_model
  interactive_markers
  pcl_ros
)


find_package(Theia REQUIRED)
include_directories(${THEIA_INCLUDE_DIRS})



#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -DEIGEN_DONT_PARALLELIZE")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem)

find_package(Ceres REQUIRED)
include_directories(  ${CERES_INCLUDE_DIRS} )


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gmm_pointcloud_align
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OPENCV_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/gmm_pointcloud_align.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/gmm_pointcloud_align_node.cpp)


FILE( GLOB KuseUtils
    src/utils/PoseManipUtils.cpp
    src/utils/RosMarkerUtils.cpp
    src/utils/MeshObject.cpp
    src/utils/RawFileIO.cpp
    src/utils/MiscUtils.cpp
    src/utils/CameraGeometry.cpp
    src/utils/PointFeatureMatching.cpp
    src/utils/GMSMatcher/gms_matcher.cpp
)

FILE( GLOB KaixuanUtils
    src/surfel_fusion/surfel_map.cpp
    src/surfel_fusion/fusion_functions.cpp
    )

FILE( GLOB GaussUtils
    src/utils/GaussianMixtureDataGenerator.cpp
    src/utils/GaussianFunction.cpp
)


FILE(GLOB CamodocalCameraModelSources
        src/utils/camodocal/src/chessboard/Chessboard.cc
        src/utils/camodocal/src/calib/CameraCalibration.cc
        src/utils/camodocal/src/camera_models/Camera.cc
        src/utils/camodocal/src/camera_models/CameraFactory.cc
        src/utils/camodocal/src/camera_models/CostFunctionFactory.cc
        src/utils/camodocal/src/camera_models/PinholeCamera.cc
        src/utils/camodocal/src/camera_models/CataCamera.cc
        src/utils/camodocal/src/camera_models/EquidistantCamera.cc
        src/utils/camodocal/src/camera_models/ScaramuzzaCamera.cc
        src/utils/camodocal/src/sparse_graph/Transform.cc
        src/utils/camodocal/src/gpl/gpl.cc
        src/utils/camodocal/src/gpl/EigenQuaternionParameterization.cc
    )


option(COMPILE_UNIT_TEST "Compile unit tests" OFF) #OFF by default. catkin_make -DCOMPILE_UNIT_TEST=ON will compile it. OFF will skip it

if(COMPILE_UNIT_TEST)

add_executable( test_objloader
    src/test_objloader.cpp
    ${KuseUtils}
    )

    target_link_libraries( test_objloader
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )

add_executable( test_multivariate_gaussian
    src/test_multivariate_gaussian.cpp
    ${KuseUtils}
    ${GaussUtils}
    )

    target_link_libraries( test_multivariate_gaussian
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )

add_executable( gmm_1d
    src/gmm_1d.cpp
    src/GMMFit.cpp
    ${KuseUtils}
    ${GaussUtils}
    )

    target_link_libraries( gmm_1d
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )


add_executable( gmm_2d
    src/gmm_2d.cpp
    src/GMMFit.cpp
    ${KuseUtils}
    ${GaussUtils}
    )
    target_link_libraries( gmm_2d
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )

add_executable( gmm_realobj
    src/gmm_realobj.cpp
    ${KuseUtils}
    ${GaussUtils}
    src/GMMFit.cpp
    )

       target_link_libraries( gmm_realobj
          ${catkin_LIBRARIES}
          ${OpenCV_LIBS}
        )

add_executable( local_ptcloud
    src/local_ptcloud.cpp
    ${CamodocalCameraModelSources}
    ${KuseUtils}
    src/SlicClustering.cpp
    src/SurfelMap.cpp
    )

        target_link_libraries( local_ptcloud
           ${catkin_LIBRARIES}
           ${OpenCV_LIBS}
         )



    add_executable( icl_nuim_tool
        src/icl_nuim_tool.cpp
        src/ICLNUIMLoader.cpp
        ${CamodocalCameraModelSources}
        ${KuseUtils}
        ${KaixuanUtils}
        )


              target_link_libraries( icl_nuim_tool
                 ${catkin_LIBRARIES}
                 ${OpenCV_LIBS}
                 #${THEIA_LIBRARIES}
               )


    add_executable( densesurfelmapping_standalone
        #src/surfel_fusion/main.cpp
        src/densesurfelmapping_standalone.cpp
        ${KuseUtils}
        ${KaixuanUtils}
    )


          target_link_libraries( densesurfelmapping_standalone
             ${catkin_LIBRARIES}
             ${OpenCV_LIBS}
           )

add_executable( inspect_data
    src/inspect_data.cpp
    )

    target_link_libraries( inspect_data
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )

    add_executable( test_inspect_checkpoint
        src/test_inspect_checkpoint.cpp
        ${KuseUtils}
        )

              target_link_libraries( test_inspect_checkpoint
                 ${catkin_LIBRARIES}
                 ${OpenCV_LIBS}
               )

add_executable( test_slic
    src/test_slic.cpp
    ${KuseUtils}
    src/SlicClustering.cpp
    )

          target_link_libraries( test_slic
             ${catkin_LIBRARIES}
             ${OpenCV_LIBS}
           )

add_executable( mean_of_vector_using_hubernorm
    src/mean_of_vector_using_hubernorm.cpp
    )


         target_link_libraries( mean_of_vector_using_hubernorm
            ${catkin_LIBRARIES}
            ${OpenCV_LIBS}
          )


#add_executable( test_nanoflann
#        src/test_nanoflann.cpp
#        )

#      target_link_libraries( test_nanoflann
#         ${catkin_LIBRARIES}
#         ${OpenCV_LIBS}
#       )

add_executable( test_posecomputation
    src/test_posecomputation.cpp
    ${KuseUtils}
    src/PoseComputation.cpp
    )



    target_link_libraries( test_posecomputation
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )



add_executable( test_lk_opticalflow
    src/test_lk_opticalflow.cpp
    ${KuseUtils}
    src/OpticalFlowTracker.cpp
)




target_link_libraries( test_lk_opticalflow
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
 )

add_executable( test_triangulation
    src/test_triangulation.cpp
#    ${KuseUtils}
    src/Triangulation.cpp
)



target_link_libraries( test_triangulation
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
 )


add_executable( test_monocular_depth
    src/test_monocular_depth.cpp
    ${KuseUtils}
    src/Triangulation.cpp
    src/OpticalFlowTracker.cpp
    src/utils/PointFeatureMatching.cpp
    )



     target_link_libraries( test_monocular_depth
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
      )



            add_executable( test_surf
              src/test_surf.cpp
              #${KuseUtils}
              #src/LocalBundle.cpp
              #src/EdgeAlignment.cpp
              )

            target_link_libraries( test_surf
                   ${catkin_LIBRARIES}
                   ${OpenCV_LIBS}
                 )

             add_executable( local_ptcloud2
                 src/local_ptcloud2.cpp
                 ${CamodocalCameraModelSources}
                 ${KuseUtils}
                 #src/SlicClustering.cpp
                 #src/SurfelMap.cpp
                 ${KaixuanUtils}
                 src/PoseComputation.cpp
                 src/Triangulation.cpp
                 src/LocalBundle.cpp
                 src/EdgeAlignment.cpp
                 )

             target_link_libraries( local_ptcloud2
                         ${catkin_LIBRARIES}
                         ${OpenCV_LIBS}
                         #${THEIA_LIBRARIES}
                       )

             add_executable( test_precise_gms
                 src/test_precise_gms.cpp
                 ${KuseUtils}
                 )

             target_link_libraries( test_precise_gms
                       ${catkin_LIBRARIES}
                       ${OpenCV_LIBS}
                     )


             add_executable( test_localbundle
               src/test_localbundle.cpp
               ${KuseUtils}
               src/EdgeAlignment.cpp
               src/LocalBundle.cpp
               ${CamodocalCameraModelSources}
               )

             target_link_libraries( test_localbundle
                    ${catkin_LIBRARIES}
                    ${OpenCV_LIBS}
                  )


              add_executable( test_edge_alignment
                src/test_edge_alignment.cpp
                ${KuseUtils}
                src/LocalBundle.cpp
                src/EdgeAlignment.cpp
                )

              target_link_libraries( test_edge_alignment
                     ${catkin_LIBRARIES}
                     ${OpenCV_LIBS}
                   )

               add_executable( test_edge_alignment_2
                 src/test_edge_alignment_2.cpp
                 ${KuseUtils}
                 src/LocalBundle.cpp
                 src/EdgeAlignment.cpp
                 )

              target_link_libraries( test_edge_alignment_2
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBS}
                    )



add_executable( test_lite_edge_alignment
  src/test_lite_edge_alignment.cpp
  ${KuseUtils}
  src/LocalBundle.cpp
  src/EdgeAlignment.cpp
  )

target_link_libraries( test_lite_edge_alignment
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )

add_executable( test_lite_gms
  src/test_lite_gms.cpp
  ${KuseUtils}
  src/LocalBundle.cpp
  src/EdgeAlignment.cpp
  )

 target_link_libraries( test_lite_gms
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
      )



add_executable( test_compare_edgealign_to_odomGT
  src/test_compare_edgealign_to_odomGT.cpp
  ${KuseUtils}
  src/LocalBundle.cpp
  src/EdgeAlignment.cpp
  )

target_link_libraries( test_compare_edgealign_to_odomGT
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
     )


 endif(COMPILE_UNIT_TEST)
 unset(COMPILE_UNIT_TEST CACHE) # <---- this is the important!!




add_executable( test_quantitative_comparison_with_pnp_vs_ea
  src/test_quantitative_comparison_with_pnp_vs_ea.cpp
  ${KuseUtils}
  src/LocalBundle.cpp
  src/EdgeAlignment.cpp
  src/DlsPnpWithRansac.cpp
  )

target_link_libraries( test_quantitative_comparison_with_pnp_vs_ea
       ${catkin_LIBRARIES}
       ${OpenCV_LIBS}
       ${THEIA_LIBRARIES}
     )


## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )



















#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gmm_pointcloud_align.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
