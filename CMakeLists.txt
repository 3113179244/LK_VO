cmake_minimum_required(VERSION 3.8)
project(LK_VO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wextra -Wpedantic -w)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
  src
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${PANGOLIN_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}_lib
  src/Camera.cpp
  src/Config.cpp
  src/FeatureDetector.cpp
  src/Frame.cpp
  src/Initializer.cpp
  src/KeyFrame.cpp
  src/Map.cpp
  src/MapPoint.cpp
  src/Optimizer.cpp
  src/System.cpp
  src/Tracker.cpp
  src/Viewer.cpp
  src/DepthFilter.cpp
)

target_link_libraries(${PROJECT_NAME}_lib
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  Sophus::Sophus
  pango_display        
  ${OPENGL_LIBRARIES}
  GLEW::GLEW
  pthread
)

add_executable(LK_VO src/main.cpp)

target_link_libraries(LK_VO ${PROJECT_NAME}_lib)

install(TARGETS ${PROJECT_NAME}_lib LK_VO
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)