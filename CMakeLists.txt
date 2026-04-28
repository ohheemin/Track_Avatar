cmake_minimum_required(VERSION 3.8)
project(avatar)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CMAKE_COMPILER_IS_CLANG)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  src/dxl_subscriber.py
  src/jerk_limit_filter.py
  src/rviz_control.py
  src/pose.py
  src/vision.py
  src/vision_nk.py
  src/clock.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  meshes
  urdf
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# C++ 헤더 파일이 있는 include 디렉토리 설치
install(DIRECTORY
  include/
  DESTINATION include
)

ament_package()
