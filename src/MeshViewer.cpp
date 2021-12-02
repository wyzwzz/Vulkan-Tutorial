//
// Created by wyz on 2021/12/2.
//
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <optional>
#include <set>
#include <cstdint>
#include <algorithm>
#include <array>
#include <unordered_map>

#include "logger.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData){
    LOG_INFO("validation layer: {0}",pCallbackData->pMessage);
    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,VkDebugUtilsMessengerEXT* pDebugMessenger){
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete(){
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex{
    glm::vec3 pos;
    glm::vec3 normal;

    bool operator==(const Vertex& other) const{
        return pos == other.pos && normal==other.normal;
    }

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,2> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex,pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex,normal);

        return attributeDescriptions;
    }
};

namespace std{
template <> struct hash<Vertex>{
    size_t operator()(const Vertex& vertex) const{
        return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ;
    }
};
}

class MeshViewerApplication{
    GLFWwindow* window;
    int window_w = 1920;
    int window_h = 1080;
    int iGPU = 0;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;//color depth
    VkExtent2D swapChainExtent;//resolution fo images in swap chain
    std::vector<VkImageView> swapChainImageViews;

    struct {
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    }mvp;


  public:
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    void initWindow(){
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);

        window = glfwCreateWindow(window_w,window_h,"MPIMeshViewer",nullptr,nullptr);

        glfwSetWindowUserPointer(window,this);
    }
    void initVulkan(){
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createSwapChainImageViews();
//        createRenderPass();
//        createDescriptorSetLayout();
//        createGraphicsPipeline();
//        createCommandPool();
//        createColorResources();
//        createDepthResources();
//        createFramebuffers();
//        createTextureImage();
//        createTextureImageView();
//        createTextureSampler();
//        loadModel();
//        createVertexBuffer();
//        createIndexBuffer();
//        createUniformBuffers();
//        createDescriptorPool();
//        createDescriptorSets();
//        createCommandBuffers();
//        createSyncObjects();
    }
    void mainLoop(){
        while(!glfwWindowShouldClose(window)){
            glfwPollEvents();
//            drawFrame();
        }
    }
    void cleanup(){


        vkDestroyDevice(device,nullptr);

        vkDestroySurfaceKHR(instance,surface,nullptr);
        if(enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(instance,debugMessenger, nullptr);
        }
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

  private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createSwapChainImageViews();
  private:
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSampleCountFlagBits getMaxUsableSampleCount();
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkImageView createImageView(VkImage image,VkFormat format,VkImageAspectFlags aspectFlags,uint32_t mipLevels);
};
void MeshViewerApplication::createInstance()
{
    if(enableValidationLayers && !checkValidationLayerSupport()){
        throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "MPIMeshViewer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    {
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        LOG_INFO("available extensions:");
        for (const auto &extension : extensions)
        {
            LOG_INFO("\t{0}",extension.extensionName);
        }
    }

    auto extensions = getRequiredExtensions();

    createInfo.enabledExtensionCount = extensions.size();
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

    if(enableValidationLayers){
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext= (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else{
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if(vkCreateInstance(&createInfo,nullptr,&instance) != VK_SUCCESS){
        throw std::runtime_error("failed to create instance!");
    }
}
bool MeshViewerApplication::checkValidationLayerSupport()
{
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount,nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount,availableLayers.data());

    LOG_INFO("available validation layer:");
    for(int i=0;i<layerCount;i++){
        LOG_INFO("\t{0}",availableLayers[i].layerName);
    }

    for(const char* layerName:validationLayers){
        bool layerFound = false;
        for(const auto& layerProperties:availableLayers){
            if(strcmp(layerName,layerProperties.layerName) == 0){
                layerFound = true;
                break;
            }
        }
        if(!layerFound){
            return false;
        }
    }
    return true;
}
std::vector<const char *> MeshViewerApplication::getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    LOG_INFO("glfwExtensionCount: {0}",glfwExtensionCount);
    for(int i = 0;i<glfwExtensionCount;i++){
        LOG_INFO("glfwExtensions: {0}",*(glfwExtensions+i));
    }
    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}
void MeshViewerApplication::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                 | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}
void MeshViewerApplication::setupDebugMessenger()
{
    if(!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);
    if(CreateDebugUtilsMessengerEXT(instance,&createInfo,nullptr,&debugMessenger)!=VK_SUCCESS){
        throw std::runtime_error("failed to set up debug messenger!");
    }
}
void MeshViewerApplication::createSurface()
{
    if(glfwCreateWindowSurface(instance,window,nullptr,&surface)!=VK_SUCCESS){
        throw std::runtime_error("failed to create window surface!");
    }
}
void MeshViewerApplication::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance,&deviceCount,nullptr);

    if(deviceCount == 0){
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance,&deviceCount,devices.data());

    LOG_INFO("physical device count {0}",deviceCount);
    for(int i=0;i<devices.size();i++){
        const auto& device = devices[i];
        if(i==iGPU && isDeviceSuitable(device)){
            physicalDevice = device;
            msaaSamples = getMaxUsableSampleCount();
            break;
        }
    }
    if(physicalDevice == VK_NULL_HANDLE){
        throw std::runtime_error("GPU " + std::to_string(iGPU)+" is not found or suitable!");
    }
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice,&deviceProperties);
    LOG_INFO("physical device in use {0}:{1}",deviceProperties.deviceID,deviceProperties.deviceName);
}
bool MeshViewerApplication::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if(extensionsSupported){
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures;
    vkGetPhysicalDeviceFeatures(device,&supportedFeatures);

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
}
QueueFamilyIndices MeshViewerApplication::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,nullptr);

    LOG_INFO("queue family count: {0}",queueFamilyCount);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,queueFamilies.data());

    int i = 0;
    for(const auto& queueFamily:queueFamilies){
        if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT){
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device,i,surface,&presentSupport);
        if(presentSupport){
            indices.presentFamily = i;
        }
        if(indices.isComplete()){
            break;
        }
        i++;
    }
    return indices;
}
bool MeshViewerApplication::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device,nullptr,
                                         &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device,nullptr,
                                         &extensionCount,availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(),
                                             deviceExtensions.end());

    for(const auto& extension: availableExtensions){
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}
SwapChainSupportDetails MeshViewerApplication::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device,surface,&details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device,surface,&formatCount, nullptr);
    if(formatCount!=0){
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device,surface,&formatCount,details.formats.data());
    }

    LOG_INFO("physical device surface format count: {0}",formatCount);

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device,surface,&presentModeCount, nullptr);
    if(presentModeCount!=0){
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device,surface,&presentModeCount,
                                                  details.presentModes.data());
    }

    LOG_INFO("physical device surface present mode count: {0}",presentModeCount);

    return details;
}
VkSampleCountFlagBits MeshViewerApplication::getMaxUsableSampleCount()
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice,&physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
                                & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if(counts & VK_SAMPLE_COUNT_64_BIT){
        LOG_INFO("msaa 64");
        return VK_SAMPLE_COUNT_64_BIT;
    }
    if(counts & VK_SAMPLE_COUNT_32_BIT){
        LOG_INFO("msaa 32");
        return VK_SAMPLE_COUNT_32_BIT;
    }
    if(counts & VK_SAMPLE_COUNT_16_BIT){
        LOG_INFO("msaa 16");
        return VK_SAMPLE_COUNT_16_BIT;
    }
    if(counts & VK_SAMPLE_COUNT_8_BIT){
        LOG_INFO("msaa 8");
        return VK_SAMPLE_COUNT_8_BIT;
    }
    if(counts & VK_SAMPLE_COUNT_4_BIT){
        LOG_INFO("msaa 4");
        return VK_SAMPLE_COUNT_4_BIT;
    }
    if(counts & VK_SAMPLE_COUNT_2_BIT){
        LOG_INFO("msaa 2");
        return VK_SAMPLE_COUNT_2_BIT;
    }
    LOG_INFO("msaa 1");
    return VK_SAMPLE_COUNT_1_BIT;
}
void MeshViewerApplication::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};
    float queuePriority = 1.f;

    for(uint32_t queueFamily:uniqueQueueFamilies){
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex=queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures  deviceFeatures{};
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    if(enableValidationLayers){
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else{
        createInfo.enabledLayerCount = 0;
    }
    if(vkCreateDevice(physicalDevice,&createInfo, nullptr,&device) != VK_SUCCESS){
        throw std::runtime_error("failed to create logical device!");
    }
    vkGetDeviceQueue(device,indices.graphicsFamily.value(),0,&graphicsQueue);
    vkGetDeviceQueue(device,indices.presentFamily.value(),0,&presentQueue);
}
void MeshViewerApplication::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if(swapChainSupport.capabilities.maxImageCount>0 && imageCount> swapChainSupport.capabilities.maxImageCount){
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    LOG_INFO("swap chain image count {}",imageCount);

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),
                                     indices.presentFamily.value()};

    if(indices.graphicsFamily != indices.presentFamily){
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else{
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if(vkCreateSwapchainKHR(device,&createInfo,nullptr,&swapChain)!= VK_SUCCESS){
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device,swapChain,&imageCount, nullptr);
    LOG_INFO("created image count: {}",imageCount);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device,swapChain,&imageCount,swapChainImages.data());

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}
VkSurfaceFormatKHR MeshViewerApplication::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
    for(const auto& availableFormat:availableFormats){
        if(availableFormat.format==VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
            return availableFormat;
        }
    }
    return availableFormats[0];
}
VkPresentModeKHR MeshViewerApplication::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes)
{
    for(auto& availablePresentMode:availablePresentModes){
        if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D MeshViewerApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if(capabilities.currentExtent.width != UINT32_MAX){
        return capabilities.currentExtent;
    }
    else{
        int width,height;
        glfwGetFramebufferSize(window,&width,&height);
        VkExtent2D actualExtent={
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        actualExtent.width = std::clamp(actualExtent.width,capabilities.minImageExtent.width,
                                        capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height,capabilities.minImageExtent.height,
                                         capabilities.maxImageExtent.height);
        return actualExtent;
    }
}
void MeshViewerApplication::createSwapChainImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());
    for(size_t i =0;i<swapChainImages.size();i++){
        swapChainImageViews[i] = createImageView(swapChainImages[i],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT,1);
    }
}
VkImageView MeshViewerApplication::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags,
                                                   uint32_t mipLevels)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if(vkCreateImageView(device,&viewInfo, nullptr,&imageView)!=VK_SUCCESS){
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

int main(){
    MeshViewerApplication app;
    try{
        app.run();
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("program exit with error {}",err.what());
        return -1;
    }
    return 0;
}