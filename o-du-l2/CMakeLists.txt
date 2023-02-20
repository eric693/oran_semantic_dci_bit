cmake_minimum_required(VERSION 3.16)


project(channel)
# 輸出的執行檔
add_executable(channel channel.c)



set(Python_ADDITIONAL_VERSIONS 3.8)
find_package(PythonLibs 3.8 REQUIRED) #3.12 is deprecated
find_package(Python 3.8 COMPONENTS Interpreter Development)
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARIES}) 



include_directories(${PYTHON_INCLUDE_DIRS})

#project(encoder)
#find_package(Python3 COMPONENTS Development)
#add_executable(encoder encoder.c)
#target_include_directories(encoder PUBLIC ${Python3_INCLUDE_DIRS})
#target_link_libraries(encoder LINK_PUBLIC ${Python3_LIBRARIES})

project(input_data)
find_package(Python3 COMPONENTS Development)
add_executable(input_data input_data.c)
target_include_directories(input_data PUBLIC ${Python3_INCLUDE_DIRS})
target_link_libraries(input_data LINK_PUBLIC ${Python3_LIBRARIES})



#include_directories(/usr/include/python3.8/Python.h)

set(PYTHON_EXECUTABLE "/usr/bin/python3.8")
set(PYTHON_INCLUDE_DIR "/usr/include/python3.8")
set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.8.so")

#gcc encoder.c -I/usr/include/python3.8 -lpython3.8


set (my_test

/home/eric/semantic/o-du-l2/encoder.c
/home/eric/semantic/o-du-l2/channel.c 
/home/eric/semantic/o-du-l2/decoder.c 
/home/eric/semantic/o-du-l2/input_data.c
/home/eric/semantic/o-du-l2/candidate_sentence.c 



 )

add_library(TEST
  ${my_test}
  )

include_directories(/home/eric/semantic/o-du-l2/src/5gnrmac)

# mkdir build 
# cd build
# cmake ..
# make -j4
# ./channel


#project(channel)

# 輸出的執行檔
#add_executable(channel channel.c) 
# 確認需要連結的 module 存在
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
# 連結到 libaries
#target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})

# 確認需要連結的 module 存在
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
#target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
# 連結到 libaries
#target_link_libraries(${PROJECT_NAME} ${Python3_LIBRARIES})
#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

#find_package(Python3 COMPONENTS Interpreter Development)
#message("Python_FOUND:${Python_FOUND}")
#message("Python_VERSION:${Python_VERsSION}")
#message("Python_Development_FOUND:${Python_Development_FOUND}")
#message("Python_LIBRARIES:${Python_LIBRARIES}")
#find_package(PythonLibs REQUIRED)