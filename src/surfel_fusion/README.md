Wang Kaixuan's Dense Surfel Mapping code resides here.

The functional code is almost the same, the input interface has been changed.



## CMake
```
find_package(catkin REQUIRED COMPONENTS
bla bla
  pcl_ros
)



FILE( GLOB KaixuanUtils
    src/surfel_fusion/surfel_map.cpp
    src/surfel_fusion/fusion_functions.cpp
    )



add_executable( kai_main
    src/surfel_fusion/main.cpp
    ${KaixuanUtils}
)


target_link_libraries( kai_main
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
 )
```
