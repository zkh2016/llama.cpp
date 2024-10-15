set(OpenCV_DIR "D:\\project\\OpenCV\\opencv\\build")
find_package(OpenCV REQUIRED)

add_library(llava OBJECT
            llava.cpp
            llava.h
            clip.cpp
            clip.h
            PillowResize.cpp
            PillowResize.hpp
            )

target_link_libraries(llava PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(llava PUBLIC ${OpenCV_LIBS})

target_include_directories(llava PUBLIC .)
target_include_directories(llava PUBLIC ../..)
target_include_directories(llava PUBLIC ../../common)
target_include_directories(llava PUBLIC ${OpenCV_INCLUDE_DIRS})

target_compile_features(llava PRIVATE cxx_std_11)

add_library(llava_static STATIC $<TARGET_OBJECTS:llava>)
if (BUILD_SHARED_LIBS)
    set_target_properties(llava PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llava PRIVATE LLAMA_SHARED LLAMA_BUILD)
    add_library(llava_shared SHARED $<TARGET_OBJECTS:llava>)
    target_link_libraries(llava_shared PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS llava_shared LIBRARY)
endif()

if (NOT MSVC)
    target_compile_options(llava PRIVATE -Wno-cast-qual) # stb_image.h
endif()

if(TARGET BUILD_INFO)
    add_dependencies(llava BUILD_INFO)
endif()

set(TARGET llava-cli)
add_executable(llava-cli llava-cli.cpp)
install(TARGETS llava-cli RUNTIME)
target_link_libraries(llava-cli PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llava PRIVATE cxx_std_11)

set(TARGET minicpmv-cli)
add_executable(minicpmv-cli minicpmv-cli.cpp)
install(TARGETS minicpmv-cli RUNTIME)
target_link_libraries(minicpmv-cli PRIVATE common llava ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llava PRIVATE cxx_std_11)
