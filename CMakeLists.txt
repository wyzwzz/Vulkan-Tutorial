cmake_minimum_required(VERSION 3.16)
project(Vulkan-Tutorial LANGUAGES CXX C)

find_library(Vulkan_LIBRARY NAMES vulkan-1 vulkan PATHS ${PROJECT_SOURCE_DIR}/third_party/vulkan/lib)
set(Vulkan_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/vulkan/include)


add_subdirectory(src)