include(${PROJECT_SOURCE_DIR}/third_party/glfw.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/glm.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/tinyobjloader.cmake)
include(${PROJECT_SOURCE_DIR}/third_party/spdlog.cmake)
add_executable(tutorial_1 tutorial_1.cpp)


target_link_libraries(tutorial_1 PRIVATE
        ${Vulkan_LIBRARY}
        glfw
        glm
        tinyobjloader
        )
target_include_directories(tutorial_1
        PRIVATE
        ${Vulkan_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/third_party
        )
target_compile_features(tutorial_1
        PRIVATE
        cxx_std_17)

add_executable(MeshViewer MeshViewer.cpp)

target_link_libraries(MeshViewer PRIVATE
        ${Vulkan_LIBRARY}
        glfw
        glm
        tinyobjloader
        spdlog::spdlog
        )
target_include_directories(MeshViewer
        PRIVATE
        ${Vulkan_INCLUDE_DIR}
        )
target_compile_features(MeshViewer PRIVATE
        cxx_std_17)
