include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)

add_executable(tutorial_1 tutorial_1.cpp)
target_link_libraries(tutorial_1 PRIVATE
        ${Vulkan_LIBRARY}
        glfw
        glm
        )
target_include_directories(tutorial_1
        PRIVATE
        ${Vulkan_INCLUDE_DIR})
target_compile_features(tutorial_1
        PRIVATE
        cxx_std_17)