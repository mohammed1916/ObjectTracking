cmake_minimum_required(VERSION 3.5)
project(ObjectTracking)

set(CMAKE_CXX_STANDARD 17)

add_executable (ObjectTracking main.cpp)

set(OpenCV_DIR "C:/Users/abd/d/ai/opencv_build/install/x64/vc17/lib")
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
  target_include_directories(ObjectTracking PUBLIC
    ${OpenCV_INCLUDE_DIRS})
  target_link_libraries (ObjectTracking ${OpenCV_LIBS})

  get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG) 
  get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

  add_custom_command(TARGET ObjectTracking POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different           
    "$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>"      
    $<TARGET_FILE_DIR:ObjectTracking>)
endif()