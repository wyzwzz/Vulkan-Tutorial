//
// Created by wyz on 2021/12/6.
//
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <json.hpp>

#include "camera.hpp"
#include "logger.hpp"
#include "transfer_function.hpp"

#include <optional>
#include <string>
#include <set>
#include <array>
#include <fstream>

const std::string assetPath = "C:/Users/wyz/projects/Vulkan-Tutorial/data/";

const std::string modelPath = "E:/neurons/neurons.json";

const int MAX_FRAMES_IN_FLIGHT = 1;

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
    Vertex(const glm::vec3& p):pos(p){}
    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,1> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,1> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex,pos);

        return attributeDescriptions;
    }
};

class VolumeViewerApplication{
    GLFWwindow* window;
    int window_w = 1920;
    int window_h = 1080;
    int iGPU = 0;
    std::unique_ptr<control::Camera> camera;
    //---------------------------------------------------------------------------

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

    //---------------------------------------------------------------------------
    struct FrameBufferAttachment{
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format;
    };
    FrameBufferAttachment rayEntry,rayExit;

    //single render pass but two subpass
    VkRenderPass renderPass;

    struct{
        VkDescriptorSetLayout rayPos;
        VkDescriptorSetLayout rayCast;
    } descriptorSetLayouts;

    struct{
        VkPipelineLayout rayPos;
        VkPipelineLayout rayCast;
    } pipelineLayouts;
    //volume render use two graphics pipelines
    struct{
        VkPipeline rayPos;
        VkPipeline rayCast;
    } graphicsPipelines;

    VkCommandPool commandPool;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    //---------------------------------------------------------------------------

    struct{
      std::vector<Vertex> vertices;
      std::vector<uint32_t> indices;
      VkBuffer vertexBuffer;
      VkDeviceMemory vertexBufferMemory;
      VkBuffer indexBuffer;
      VkDeviceMemory indexBufferMemory;
    } proxyCube;

    VkDescriptorPool descriptorPool;

    //for each swap chain image in order to multi-thread draw
    struct{
        std::vector<VkDescriptorSet> rayPos;
        std::vector<VkDescriptorSet> rayCast;
    } descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    struct{
        VkImage image;
        VkDeviceMemory mem;
        VkImageView view;
        VkSampler sampler;
    } transferFunc,volumeData;

    float volume_space_x = 0.01f,volume_space_y = 0.01f,volume_space_z = 0.01f;

    //upload once before render
    struct VolumeInfo{
        float volume_board_x;
        float volume_board_y;
        float volume_board_z;
        float step;
    }volumeInfo;

    //update every frame
    struct MVP{
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };

    struct UBO{
        VkBuffer buffer;
        VkDeviceMemory bufferMemory;
    };
    std::vector<UBO> mvpUBO;
    std::vector<UBO> viewPosUBO;
    std::vector<UBO> volumeInfoUBO;

    //may parallel render image in gpu
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;
    size_t currentFrame = 0;
  public:
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

  private:
    void initWindow(){
        //create window
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);

        window = glfwCreateWindow(window_w,window_h,"VolumeViewer",nullptr,nullptr);

        glfwSetWindowUserPointer(window,this);

        //register event callback
        registerEventCallback();
    }
    void initVulkan(){
        //1. default vulkan resources
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createSwapChainImageViews();
        //2.draw static vulkan resources
        createFramebufferAttachments();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createDepthResource();
        createFramebuffers();
        //3.draw dynamic vulkan resources
        loadVolume();
        createProxyCubeResource();
        createUBOResource();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }
    void mainLoop(){
        while(!glfwWindowShouldClose(window)){
            glfwPollEvents();
            drawFrame();
        }
    }
    void cleanup(){

    }

  private:
    inline static std::function<void(GLFWwindow*,int,int,int,int)> KeyCallback;
    inline static std::function<void(GLFWwindow*,double,double)> ScrollCallback;
    inline static std::function<void(GLFWwindow*,double,double)> CursorPosCallback;
    inline static std::function<void(GLFWwindow*,int,int,int)> MouseButtonCallback;
    inline static void glfw_key_callback(GLFWwindow* window,int key,int scancode,int action,int mods){
        KeyCallback(window,key,scancode,action,mods);
    }
    inline static void glfw_scroll_callback(GLFWwindow* window,double xoffset,double yoffset){
        ScrollCallback(window,xoffset,yoffset);
    }
    inline static void glfw_cursor_pos_callback(GLFWwindow* window,double xpos,double ypos){
        CursorPosCallback(window,xpos,ypos);
    }
    inline static void glfw_mouse_button_callback(GLFWwindow* window,int button,int action,int mods){
        MouseButtonCallback(window,button,action,mods);
    }
    void registerEventCallback();
  private:
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createSwapChainImageViews();

    void createFramebufferAttachments();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createCommandPool();
    void createDepthResource();
    void createFramebuffers();

    void loadVolume();
    void createProxyCubeResource();
    void createUBOResource();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

    void drawFrame();
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
    void createImage(uint32_t width,uint32_t height,uint32_t mipLevels,
                     VkSampleCountFlagBits numSamples,VkFormat format,
                     VkImageTiling tiling,VkImageUsageFlags usage,VkMemoryPropertyFlags properties,
                     VkImage& image,VkDeviceMemory& imageMemory);
    uint32_t findMemoryType(uint32_t typeFilter,VkMemoryPropertyFlags properties);
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,VkImageTiling tiling,VkFormatFeatureFlags features);
    static std::vector<char> readFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<uint8_t> readVolumeFile(const std::string& filename);
    void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,VkDeviceMemory& bufferMemory);
    void transitionImageLayout(VkImage image,VkFormat,VkImageLayout oldLayout,VkImageLayout newLayout);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer);
    void loadTransferFunc(const std::string& filename);
    void createVertexBuffer(VkBuffer& vertexBuffer,VkDeviceMemory& vertexBufferMemory,const std::vector<Vertex>& vertices);
    void createIndexBuffer(VkBuffer& indexBuffer,VkDeviceMemory& indexBufferMemory,const std::vector<uint32_t>& indices);
    void copyBuffer(VkBuffer srcBuffer,VkBuffer dstBuffer,VkDeviceSize size);
    void updateUniformBuffer(uint32_t currentImage);
};
void VolumeViewerApplication::registerEventCallback()
{
    camera = std::make_unique<control::TrackBallCamera>(1.28f,window_w,window_h,glm::vec3{1.28f,1.28f,1.28f});
    ScrollCallback = [&](GLFWwindow* window,double xoffset,double yoffset){
      camera->processMouseScroll(yoffset);
    };
    static bool left_mouse_press = false;
    MouseButtonCallback = [this](GLFWwindow* window,int button,int action,int mods)->void{
      if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
          left_mouse_press = true;
          double xpos,ypos;
          glfwGetCursorPos(window,&xpos,&ypos);
          camera->processMouseButton(control::CameraDefinedMouseButton::Left,true,xpos,ypos);
      }
      else if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
          left_mouse_press = false;
          camera->processMouseButton(control::CameraDefinedMouseButton::Left,false,0.0,0.0);
      }
    };
    CursorPosCallback = [&](GLFWwindow* window,double xpos,double ypos)->void{
      if(left_mouse_press){
          camera->processMouseMove(xpos,ypos);
      }
    };
    glfwSetScrollCallback(window,glfw_scroll_callback);
    glfwSetCursorPosCallback(window,glfw_cursor_pos_callback);
    glfwSetMouseButtonCallback(window,glfw_mouse_button_callback);

}
void VolumeViewerApplication::createInstance()
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
bool VolumeViewerApplication::checkValidationLayerSupport()
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
std::vector<const char *> VolumeViewerApplication::getRequiredExtensions()
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
void VolumeViewerApplication::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                 | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}
void VolumeViewerApplication::setupDebugMessenger()
{
    if(!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);
    if(CreateDebugUtilsMessengerEXT(instance,&createInfo,nullptr,&debugMessenger)!=VK_SUCCESS){
        throw std::runtime_error("failed to set up debug messenger!");
    }
}
void VolumeViewerApplication::createSurface()
{
    if(glfwCreateWindowSurface(instance,window,nullptr,&surface)!=VK_SUCCESS){
        throw std::runtime_error("failed to create window surface!");
    }
}
void VolumeViewerApplication::pickPhysicalDevice()
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
bool VolumeViewerApplication::isDeviceSuitable(VkPhysicalDevice device)
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
QueueFamilyIndices VolumeViewerApplication::findQueueFamilies(VkPhysicalDevice device)
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
bool VolumeViewerApplication::checkDeviceExtensionSupport(VkPhysicalDevice device)
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
SwapChainSupportDetails VolumeViewerApplication::querySwapChainSupport(VkPhysicalDevice device)
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
VkSampleCountFlagBits VolumeViewerApplication::getMaxUsableSampleCount()
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
void VolumeViewerApplication::createLogicalDevice()
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
void VolumeViewerApplication::createSwapChain()
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
VkSurfaceFormatKHR VolumeViewerApplication::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
    for(const auto& availableFormat:availableFormats){
        if(availableFormat.format==VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
            return availableFormat;
        }
    }
    return availableFormats[0];
}
VkPresentModeKHR VolumeViewerApplication::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes)
{
    for(auto& availablePresentMode:availablePresentModes){
        if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
            return availablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D VolumeViewerApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
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
void VolumeViewerApplication::createSwapChainImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());
    for(size_t i =0;i<swapChainImages.size();i++){
        swapChainImageViews[i] = createImageView(swapChainImages[i],swapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT,1);
    }
}
VkImageView VolumeViewerApplication::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags,
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
void VolumeViewerApplication::createFramebufferAttachments()
{
    //rayEntry attachment
    {
        rayEntry.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        createImage(swapChainExtent.width,swapChainExtent.height,
                    1,VK_SAMPLE_COUNT_1_BIT,
                    rayEntry.format,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    rayEntry.image,rayEntry.mem);
        rayEntry.view = createImageView(rayEntry.image,rayEntry.format,
                                        VK_IMAGE_ASPECT_COLOR_BIT,1);
    }
    //rayExit attachment
    {
        rayExit.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        createImage(swapChainExtent.width,swapChainExtent.height,
                    1,VK_SAMPLE_COUNT_1_BIT,
                    rayExit.format,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    rayExit.image,rayExit.mem);
        rayExit.view = createImageView(rayExit.image,rayExit.format,
                                       VK_IMAGE_ASPECT_COLOR_BIT,1);
    }
}
void VolumeViewerApplication::createRenderPass()
{
    std::array<VkAttachmentDescription,4> attachments{};

    //0 for rayEntry
    attachments[0].format = rayEntry.format;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    //1 for rayExit
    attachments[1].format = rayExit.format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    //2 for rayCast
    attachments[2].format = swapChainImageFormat;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    //3 for depth
    attachments[3].format = findDepthFormat();
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    // two subpasses
    std::array<VkSubpassDescription,2> subpass{};

    // first subpass: get ray entry and exit pos
    VkAttachmentReference colorRefs[2];
    colorRefs[0] = {0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorRefs[1] = {1,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference depthRef = {3,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    subpass[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass[0].colorAttachmentCount = 2;
    subpass[0].pColorAttachments = colorRefs;
    subpass[0].pDepthStencilAttachment = &depthRef;
    subpass[0].inputAttachmentCount = 0;
    subpass[0].pInputAttachments = nullptr;
    // second pass: ray cast

    VkAttachmentReference colorRef = {2,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference inputRefs[2];
    inputRefs[0] = {0,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    inputRefs[1] = {1,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    subpass[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass[1].colorAttachmentCount = 1;
    subpass[1].pColorAttachments = &colorRef;
    subpass[1].pDepthStencilAttachment = &depthRef;
    subpass[1].inputAttachmentCount = 2;
    subpass[1].pInputAttachments = inputRefs;

    std::array<VkSubpassDependency,3> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = 1;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[2].srcSubpass = 1;
    dependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[2].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[2].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = static_cast<uint32_t>(subpass.size());
    renderPassInfo.pSubpasses = subpass.data();
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    if(vkCreateRenderPass(device,&renderPassInfo, nullptr,&renderPass)!=VK_SUCCESS){
        throw std::runtime_error("failed to create render pass!");
    }
}
void VolumeViewerApplication::createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                                          VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling,
                                          VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
                                          VkDeviceMemory &imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = numSamples;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;//optional

    if(vkCreateImage(device,&imageInfo, nullptr,&image)!=VK_SUCCESS){
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device,image,&memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties);

    if(vkAllocateMemory(device,&allocInfo,nullptr,&imageMemory)!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate image memory!");
    }
    vkBindImageMemory(device,image,imageMemory,0);
}
uint32_t VolumeViewerApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice,&memProperties);

    for(uint32_t i = 0;i<memProperties.memoryTypeCount;i++){
        if((typeFilter & (1<<i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties){
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}
VkFormat VolumeViewerApplication::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}
VkFormat VolumeViewerApplication::findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling,
                                                      VkFormatFeatureFlags features)
{
    for(VkFormat format:candidates){
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice,format,&props);

        if(tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features){
            return format;
        }
        else if(tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features){
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}
void VolumeViewerApplication::createDescriptorSetLayout()
{
    //1. ray pos
    {
        VkDescriptorSetLayoutBinding mvpLayoutBingding{};
        mvpLayoutBingding.binding = 0;
        mvpLayoutBingding.descriptorCount = 1;
        mvpLayoutBingding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mvpLayoutBingding.pImmutableSamplers = nullptr;
        mvpLayoutBingding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding viewPosBinding{};
        viewPosBinding.binding = 1;
        viewPosBinding.descriptorCount = 1;
        viewPosBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        viewPosBinding.pImmutableSamplers = nullptr;
        viewPosBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding bindings[2] = {mvpLayoutBingding, viewPosBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayouts.rayPos) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
    //2. ray cast
    {
        VkDescriptorSetLayoutBinding rayEntryInputBinding{};
        rayEntryInputBinding.binding = 0;
        rayEntryInputBinding.descriptorCount = 1;
        rayEntryInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        rayEntryInputBinding.pImmutableSamplers = nullptr;
        rayEntryInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding rayExitInputBinding{};
        rayExitInputBinding.binding = 1;
        rayExitInputBinding.descriptorCount = 1;
        rayExitInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        rayExitInputBinding.pImmutableSamplers = nullptr;
        rayExitInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding transferFuncBinding{};
        transferFuncBinding.binding = 2;
        transferFuncBinding.descriptorCount = 1;
        transferFuncBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        transferFuncBinding.pImmutableSamplers = nullptr;
        transferFuncBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding volumeDataBinding{};
        volumeDataBinding.binding = 3;
        volumeDataBinding.descriptorCount = 1;
        volumeDataBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        volumeDataBinding.pImmutableSamplers = nullptr;
        volumeDataBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding volumeInfoBinding{};
        volumeInfoBinding.binding = 4;
        volumeInfoBinding.descriptorCount = 1;
        volumeInfoBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        volumeInfoBinding.pImmutableSamplers = nullptr;
        volumeInfoBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding bindings[5] ={rayEntryInputBinding,rayExitInputBinding,
                                                    transferFuncBinding,volumeDataBinding,volumeInfoBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 5;
        layoutInfo.pBindings = bindings;
        if(vkCreateDescriptorSetLayout(device,&layoutInfo, nullptr,&descriptorSetLayouts.rayCast)!=VK_SUCCESS){
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }
}
void VolumeViewerApplication::createGraphicsPipeline()
{
    // need to create two pipelines
    // first create ray pos pipeline
    {
        auto vertShaderCode = readFile(assetPath+"shaders/VolumeViewer/ray_pos.vert.spv");
        auto fragShaderCode = readFile(assetPath+"shaders/VolumeViewer/ray_pos.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[]={vertShaderStageInfo,fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescription = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = attributeDescription.size();
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport  viewport{};
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth= 0.f;
        viewport.maxDepth = 1.f;

        VkRect2D scissor{};
        scissor.offset = {0,0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;//useful for shadow map
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;//not use cull mode
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_NEVER;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                                              | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        VkPipelineColorBlendAttachmentState colorBlendAttachments[] = {colorBlendAttachment,colorBlendAttachment};


        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 2;
        colorBlending.pAttachments = colorBlendAttachments;
        colorBlending.blendConstants[0] = 0.f;
        colorBlending.blendConstants[1] = 0.f;
        colorBlending.blendConstants[2] = 0.f;
        colorBlending.blendConstants[3] = 0.f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts.rayPos;

        if(vkCreatePipelineLayout(device,&pipelineLayoutInfo,nullptr,&pipelineLayouts.rayPos)!=VK_SUCCESS){
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayouts.rayPos;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pipelineInfo, nullptr,&graphicsPipelines.rayPos)!=VK_SUCCESS){
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device,fragShaderModule, nullptr);
        vkDestroyShaderModule(device,vertShaderModule, nullptr);
    }
    {
        auto vertShaderCode = readFile(assetPath+"shaders/VolumeViewer/ray_cast.vert.spv");
        auto fragShaderCode = readFile(assetPath+"shaders/VolumeViewer/ray_cast.frag.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[]={vertShaderStageInfo,fragShaderStageInfo};

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport  viewport{};
        viewport.x = 0.f;
        viewport.y = 0.f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth= 0.f;
        viewport.maxDepth = 1.f;

        VkRect2D scissor{};
        scissor.offset = {0,0};
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;//useful for shadow map
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                                              | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState colorBlendAttachments[] = {colorBlendAttachment};


        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = colorBlendAttachments;
        colorBlending.blendConstants[0] = 0.f;
        colorBlending.blendConstants[1] = 0.f;
        colorBlending.blendConstants[2] = 0.f;
        colorBlending.blendConstants[3] = 0.f;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayouts.rayCast;

        if(vkCreatePipelineLayout(device,&pipelineLayoutInfo,nullptr,&pipelineLayouts.rayCast)!=VK_SUCCESS){
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayouts.rayCast;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 1;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pipelineInfo, nullptr,&graphicsPipelines.rayCast)!=VK_SUCCESS){
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device,fragShaderModule, nullptr);
        vkDestroyShaderModule(device,vertShaderModule, nullptr);
    }
}
std::vector<char> VolumeViewerApplication::readFile(const std::string &filename)
{
    std::ifstream file(filename,std::ios::ate|std::ios::binary);

    if(!file.is_open()){
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(),fileSize);

    file.close();

    return buffer;
}
VkShaderModule VolumeViewerApplication::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if(vkCreateShaderModule(device,&createInfo,nullptr,&shaderModule)!=VK_SUCCESS){
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}
void VolumeViewerApplication::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if(vkCreateCommandPool(device,&poolInfo,nullptr,&commandPool)!=VK_SUCCESS){
        throw std::runtime_error("failed to create command pool!");
    }
}
void VolumeViewerApplication::createDepthResource()
{
    VkFormat depthFormat = findDepthFormat();
    createImage(swapChainExtent.width,swapChainExtent.height,
                1,VK_SAMPLE_COUNT_1_BIT,
                depthFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthImage,depthImageMemory);
    depthImageView = createImageView(depthImage,depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT,1);
}
void VolumeViewerApplication::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for(size_t i =0;i<swapChainImageViews.size();i++){
        //attachments should compatible with render pass setting
        //here are attachments instances so should compatible with render pass which already set attachments type info
        std::array<VkImageView,4> attachments = {
            rayEntry.view,rayExit.view,
            swapChainImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;
        if(vkCreateFramebuffer(device,&framebufferInfo,nullptr,&swapChainFramebuffers[i])!=VK_SUCCESS){
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}
void extractVolumeDim(int& x,int& y,int& z,const std::string& filename){
    auto _pos = filename.find_last_of('_');
    auto t = filename.substr(0,_pos);
    auto zpos = t.find_last_of('_');
    z = std::stoi(t.substr(zpos+1));
    t = t.substr(0,zpos);
    auto ypos = t.find_last_of('_');
    y = std::stoi(t.substr(ypos+1));
    t = t.substr(0,ypos);
    auto xpos = t.find_last_of('_');
    x = std::stoi(t.substr(xpos+1));

}
void VolumeViewerApplication::loadVolume()
{
    auto volume_path = assetPath+"volumes/foot_256_256_256_uint8.raw";
    auto readVolumeData = readVolumeFile(volume_path);
    int volume_dim_x,volume_dim_y,volume_dim_z;
    extractVolumeDim(volume_dim_x,volume_dim_y,volume_dim_z,volume_path);
    {
        volumeInfo.volume_board_x = volume_dim_x * volume_space_x;
        volumeInfo.volume_board_y = volume_dim_y * volume_space_y;
        volumeInfo.volume_board_z = volume_dim_z * volume_space_z;
        auto max_dim = std::max({volume_dim_x,volume_dim_y,volume_dim_z});
        volumeInfo.step = 0.3f / max_dim;
    }

    //first create buffer
    VkDeviceSize volumeSize = volume_dim_x * volume_dim_y * volume_dim_z;
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(volumeSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,stagingBufferMemory);

    void* data;
    vkMapMemory(device,stagingBufferMemory,0,volumeSize,0,&data);
    memcpy(data,readVolumeData.data(),(size_t)volumeSize);
    vkUnmapMemory(device,stagingBufferMemory);

    //create volume image
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_3D;
        imageInfo.extent.width = volume_dim_x;
        imageInfo.extent.height = volume_dim_y;
        imageInfo.extent.depth = volume_dim_z;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R8_UNORM;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = 0;

        if(vkCreateImage(device,&imageInfo,nullptr,&volumeData.image)!=VK_SUCCESS){
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device,volumeData.image,&memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if(vkAllocateMemory(device,&allocInfo,nullptr,&volumeData.mem)!=VK_SUCCESS){
            throw std::runtime_error("failed to allocate image memory!");
        }
        vkBindImageMemory(device,volumeData.image,volumeData.mem,0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = volumeData.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
        viewInfo.format = VK_FORMAT_R8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        if(vkCreateImageView(device,&viewInfo,nullptr,&volumeData.view)!=VK_SUCCESS){
            throw std::runtime_error("failed to create texture image view!");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice,&properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = 1;
        samplerInfo.mipLodBias = 0;

        if(vkCreateSampler(device,&samplerInfo, nullptr,&volumeData.sampler)!=VK_SUCCESS){
            throw std::runtime_error("failed to create texture sampler!");
        }
    }

    transitionImageLayout(volumeData.image,VK_FORMAT_R8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    //copy buffer to image
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0,0,0};
        region.imageExtent = {(uint32_t)volume_dim_x,(uint32_t)volume_dim_y,(uint32_t)volume_dim_z};

        vkCmdCopyBufferToImage(commandBuffer,stagingBuffer,volumeData.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);

        endSingleTimeCommands(commandBuffer);
    }
    transitionImageLayout(volumeData.image,VK_FORMAT_R8_UNORM,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);

    loadTransferFunc(assetPath+"volumes/foot_tf.json");
}
std::vector<uint8_t> VolumeViewerApplication::readVolumeFile(const std::string &filename)
{
    size_t volumeSize = 0;
    {
        auto _pos = filename.find_last_of('_');
        auto ext = filename.substr(_pos+1);
        if(ext != "uint8.raw"){
            throw std::runtime_error("error volume file type or format");
        }
        auto t = filename.substr(0,_pos);
        auto zpos = t.find_last_of('_');
        auto z = std::stoi(t.substr(zpos+1));
        t = t.substr(0,zpos);
        auto ypos = t.find_last_of('_');
        auto y = std::stoi(t.substr(ypos+1));
        t = t.substr(0,ypos);
        auto xpos = t.find_last_of('_');
        auto x = std::stoi(t.substr(xpos+1));
        auto name = t.substr(0,xpos);
        LOG_INFO("volume name: {0}, dim: {1} {2} {3}, ext: {4}",name,x,y,z,ext);
        volumeSize = (size_t) x * y * z;
    }
    std::ifstream in(filename,std::ios::binary|std::ios::ate);
    if(!in.is_open()){
        throw std::runtime_error("open volume file failed");
    }
    auto fileSize = in.tellg();
    if(fileSize != volumeSize){
        throw std::runtime_error("invalid volume file size");
    }
    in.seekg(0,std::ios::beg);
    std::vector<uint8_t> data(volumeSize);
    in.read(reinterpret_cast<char*>(data.data()),volumeSize);
    in.close();
    return data;
}
void VolumeViewerApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                           VkMemoryPropertyFlags properties, VkBuffer &buffer,
                                           VkDeviceMemory &bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if(vkCreateBuffer(device,&bufferInfo,nullptr,&buffer)!=VK_SUCCESS){
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device,buffer,&memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties);

    if(vkAllocateMemory(device,&allocInfo,nullptr,&bufferMemory)!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate buffer memory!");
    }
    vkBindBufferMemory(device,buffer,bufferMemory,0);
}
void VolumeViewerApplication::transitionImageLayout(VkImage image, VkFormat, VkImageLayout oldLayout,
                                                    VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else{
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer,sourceStage,destinationStage,0,
                         0, nullptr,
                         0, nullptr,
                         1,&barrier);

    endSingleTimeCommands(commandBuffer);
}
VkCommandBuffer VolumeViewerApplication::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device,&allocInfo,&commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer,&beginInfo);

    return commandBuffer;
}
void VolumeViewerApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue,1,&submitInfo,VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device,commandPool,1,&commandBuffer);
}
void VolumeViewerApplication::loadTransferFunc(const std::string &filename)
{
    std::ifstream in(filename);
    if(!in.is_open()){
        throw std::runtime_error("open file failed");
    }
    nlohmann::json j;
    in >> j;
    std::map<uint8_t,std::array<float,4>> color_map;
    auto points = j.at("tf");
    for(auto it = points.begin();it != points.end();it++){
        int key = std::stoi(it.key());
        std::array<float,4> values = it.value();
        color_map[key] = values;
    }
    TransferFunc tf(std::move(color_map));
    VkDeviceSize tfSize = sizeof(float)*TF_DIM*4;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(tfSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,stagingBufferMemory);

    void* data;
    vkMapMemory(device,stagingBufferMemory,0,tfSize,0,&data);
    memcpy(data,tf.getTransferFunction().data(),(size_t)tfSize);
    vkUnmapMemory(device,stagingBufferMemory);

    //create tf 1-d image
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_1D;
        imageInfo.extent.width = TF_DIM;
        imageInfo.extent.height = 1;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = 0;//optional

        if(vkCreateImage(device,&imageInfo,nullptr,&transferFunc.image)!=VK_SUCCESS){
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device,transferFunc.image,&memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        if(vkAllocateMemory(device,&allocInfo,nullptr,&transferFunc.mem)!=VK_SUCCESS){
            throw std::runtime_error("failed to allocate image memory!");
        }
        vkBindImageMemory(device,transferFunc.image,transferFunc.mem,0);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = transferFunc.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_1D;
        viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if(vkCreateImageView(device,&viewInfo, nullptr,&transferFunc.view)!=VK_SUCCESS){
            throw std::runtime_error("failed to create texture image view!");
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice,&properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0;
        samplerInfo.maxLod = 1;
        samplerInfo.mipLodBias = 0;

        if(vkCreateSampler(device,&samplerInfo, nullptr,&transferFunc.sampler)!=VK_SUCCESS){
            throw std::runtime_error("failed to create texture sampler!");
        }
    }
    transitionImageLayout(transferFunc.image,VK_FORMAT_R32G32B32A32_SFLOAT,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    //copy buffer to image
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0,0,0};
        region.imageExtent = {(uint32_t)TF_DIM,1,1};

        vkCmdCopyBufferToImage(commandBuffer,stagingBuffer,transferFunc.image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);

        endSingleTimeCommands(commandBuffer);
    }
    transitionImageLayout(transferFunc.image,VK_FORMAT_R32G32B32A32_SFLOAT,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
}
void VolumeViewerApplication::createProxyCubeResource()
{
    auto& vertices = proxyCube.vertices;
    vertices.emplace_back(glm::vec3{0.f,0.f,0.f});
    vertices.emplace_back(glm::vec3{volumeInfo.volume_board_x,0.f,0.f});
    vertices.emplace_back(glm::vec3{volumeInfo.volume_board_x,volumeInfo.volume_board_y,0.f});
    vertices.emplace_back(glm::vec3{0.f,volumeInfo.volume_board_y,0.f});
    vertices.emplace_back(glm::vec3{0.f,0.f,volumeInfo.volume_board_z});
    vertices.emplace_back(glm::vec3{volumeInfo.volume_board_x,0.f,volumeInfo.volume_board_z});
    vertices.emplace_back(glm::vec3{volumeInfo.volume_board_x,volumeInfo.volume_board_y,volumeInfo.volume_board_z});
    vertices.emplace_back(glm::vec3{0.f,volumeInfo.volume_board_y,volumeInfo.volume_board_z});
    auto& indices = proxyCube.indices;
    indices.emplace_back(0);indices.emplace_back(2);indices.emplace_back(1);
    indices.emplace_back(0);indices.emplace_back(3);indices.emplace_back(2);

    indices.emplace_back(0);indices.emplace_back(1);indices.emplace_back(4);
    indices.emplace_back(1);indices.emplace_back(5);indices.emplace_back(4);

    indices.emplace_back(1);indices.emplace_back(2);indices.emplace_back(5);
    indices.emplace_back(2);indices.emplace_back(6);indices.emplace_back(5);

    indices.emplace_back(2);indices.emplace_back(3);indices.emplace_back(6);
    indices.emplace_back(3);indices.emplace_back(7);indices.emplace_back(6);

    indices.emplace_back(3);indices.emplace_back(0);indices.emplace_back(7);
    indices.emplace_back(0);indices.emplace_back(4);indices.emplace_back(7);

    indices.emplace_back(4);indices.emplace_back(5);indices.emplace_back(7);
    indices.emplace_back(5);indices.emplace_back(6);indices.emplace_back(7);

    createVertexBuffer(proxyCube.vertexBuffer,proxyCube.vertexBufferMemory,proxyCube.vertices);
    createIndexBuffer(proxyCube.indexBuffer,proxyCube.indexBufferMemory,proxyCube.indices);
}
void VolumeViewerApplication::createVertexBuffer(VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory,
                                                 const std::vector<Vertex> &vertices)
{
    VkDeviceSize bufferSize = sizeof(Vertex) * vertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,stagingBufferMemory);

    void* data = nullptr;
    vkMapMemory(device,stagingBufferMemory,0,bufferSize,0,&data);
    memcpy(data,vertices.data(),(size_t)bufferSize);
    vkUnmapMemory(device,stagingBufferMemory);

    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer,vertexBufferMemory);

    copyBuffer(stagingBuffer,vertexBuffer,bufferSize);

    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
}
void VolumeViewerApplication::createIndexBuffer(VkBuffer &indexBuffer, VkDeviceMemory &indexBufferMemory,
                                                const std::vector<uint32_t> &indices)
{
    VkDeviceSize bufferSize = sizeof(uint32_t)*indices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,stagingBufferMemory);

    void* data = nullptr;
    vkMapMemory(device,stagingBufferMemory,0,bufferSize,0,&data);
    memcpy(data,indices.data(),(size_t)bufferSize);
    vkUnmapMemory(device,stagingBufferMemory);

    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 indexBuffer,indexBufferMemory);

    copyBuffer(stagingBuffer,indexBuffer,bufferSize);

    vkDestroyBuffer(device,stagingBuffer,nullptr);
    vkFreeMemory(device,stagingBufferMemory,nullptr);
}
void VolumeViewerApplication::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer,srcBuffer,dstBuffer,1,&copyRegion);

    endSingleTimeCommands(commandBuffer);
}
void VolumeViewerApplication::createUBOResource()
{
    //mvp
    {
        VkDeviceSize bufferSize = sizeof(MVP);
        mvpUBO.resize(swapChainImages.size());
        for(size_t i=0;i<swapChainImages.size();i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         mvpUBO[i].buffer,mvpUBO[i].bufferMemory);
        }
    }
    //viewPos
    {
        VkDeviceSize bufferSize = sizeof(glm::vec4);
        viewPosUBO.resize(swapChainImages.size());
        for(size_t i=0;i<swapChainImages.size();i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         viewPosUBO[i].buffer,viewPosUBO[i].bufferMemory);
        }
    }
    //volumeInfo
    {
        VkDeviceSize bufferSize = sizeof(VolumeInfo);
        volumeInfoUBO.resize(swapChainImages.size());
        for(size_t i =0;i<swapChainImages.size();i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         volumeInfoUBO[i].buffer,volumeInfoUBO[i].bufferMemory);
        }
    }
}
void VolumeViewerApplication::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize,3> poolSize{};
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size() * 3);
    poolSize[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size() * 2);
    poolSize[2].type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    poolSize[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size() * 2);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size()*2);

    if(vkCreateDescriptorPool(device,&poolInfo, nullptr,&descriptorPool)!=VK_SUCCESS){
        throw std::runtime_error("failed to create descriptor pool!");
    }
}
void VolumeViewerApplication::createDescriptorSets()
{
    //ray pos
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),descriptorSetLayouts.rayPos);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.rayPos.resize(swapChainImages.size());
        if(vkAllocateDescriptorSets(device,&allocInfo,descriptorSets.rayPos.data())!=VK_SUCCESS){
            throw std::runtime_error("failed to allocate ray pos descriptor sets!");
        }

        for(int i=0;i<swapChainImages.size();i++){
            VkDescriptorBufferInfo mvpBufferInfo{};
            mvpBufferInfo.buffer = mvpUBO[i].buffer;
            mvpBufferInfo.offset = 0;
            mvpBufferInfo.range = sizeof(MVP);

            VkDescriptorBufferInfo viewPosBufferInfo{};
            viewPosBufferInfo.buffer = viewPosUBO[i].buffer;
            viewPosBufferInfo.offset = 0;
            viewPosBufferInfo.range = sizeof(glm::vec4);

            std::array<VkWriteDescriptorSet,2> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets.rayPos[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &mvpBufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets.rayPos[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pBufferInfo = &viewPosBufferInfo;

            vkUpdateDescriptorSets(device,
                                   static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(),0,nullptr);
        }
    }
    //ray cast
    {
        std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),descriptorSetLayouts.rayCast);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.rayCast.resize(swapChainImages.size());
        int res ;
        if((res = vkAllocateDescriptorSets(device,&allocInfo,descriptorSets.rayCast.data()))!=VK_SUCCESS){
            LOG_ERROR("{}",res);
            throw std::runtime_error("failed to allocate ray cast descriptor sets!");
        }

        for(int i=0;i<swapChainImages.size();i++){

            VkDescriptorImageInfo rayEntryImageInfo{};
            rayEntryImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            rayEntryImageInfo.imageView = rayEntry.view;
            rayEntryImageInfo.sampler = VK_NULL_HANDLE;

            VkDescriptorImageInfo rayExitImageInfo{};
            rayExitImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            rayExitImageInfo.imageView = rayExit.view;
            rayExitImageInfo.sampler = VK_NULL_HANDLE;

            VkDescriptorImageInfo tfImageInfo{};
            tfImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            tfImageInfo.imageView = transferFunc.view;
            tfImageInfo.sampler = transferFunc.sampler;

            VkDescriptorImageInfo volumeImageInfo{};
            volumeImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            volumeImageInfo.imageView = volumeData.view;
            volumeImageInfo.sampler = volumeData.sampler;

            VkDescriptorBufferInfo volumeBufferInfo{};
            volumeBufferInfo.buffer = volumeInfoUBO[i].buffer;
            volumeBufferInfo.offset = 0;
            volumeBufferInfo.range = sizeof(VolumeInfo);

            std::array<VkWriteDescriptorSet,5> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets.rayCast[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &rayEntryImageInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets.rayCast[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &rayExitImageInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets.rayCast[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pImageInfo = &tfImageInfo;

            descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[3].dstSet = descriptorSets.rayCast[i];
            descriptorWrites[3].dstBinding = 3;
            descriptorWrites[3].dstArrayElement = 0;
            descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[3].descriptorCount = 1;
            descriptorWrites[3].pImageInfo = &volumeImageInfo;

            descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[4].dstSet = descriptorSets.rayCast[i];
            descriptorWrites[4].dstBinding = 4;
            descriptorWrites[4].dstArrayElement = 0;
            descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[4].descriptorCount = 1;
            descriptorWrites[4].pBufferInfo = &volumeBufferInfo;

            vkUpdateDescriptorSets(device,
                                   static_cast<uint32_t>(descriptorWrites.size()),
                                   descriptorWrites.data(),0,nullptr);
        }
    }
}
void VolumeViewerApplication::createCommandBuffers()
{
    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = (uint32_t) commandBuffers.size();
    if(vkAllocateCommandBuffers(device,&allocateInfo,commandBuffers.data())!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate command buffers!");
    }
    LOG_INFO("commandBuffers size: {}",commandBuffers.size());
    for(int i=0;i<commandBuffers.size();i++){
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if(vkBeginCommandBuffer(commandBuffers[i],&beginInfo)!=VK_SUCCESS){
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue,4> clearValues;
        clearValues[0].color = {0.f,0.f,0.f,1.f};
        clearValues[1].color = {0.f,0.f,0.f,1.f};
        clearValues[2].color = {0.f,0.f,0.f,1.f};
        clearValues[3].depthStencil = {1.f,0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[i],&renderPassInfo,VK_SUBPASS_CONTENTS_INLINE);

        {
            vkCmdBindPipeline(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,graphicsPipelines.rayPos);

            vkCmdBindDescriptorSets(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayouts.rayPos,
                                    0,1,&descriptorSets.rayPos[i],0,nullptr);

            VkDeviceSize offsets[]={0};
            vkCmdBindVertexBuffers(commandBuffers[i],0,1,&proxyCube.vertexBuffer,offsets);

            vkCmdBindIndexBuffer(commandBuffers[i],proxyCube.indexBuffer,0,VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(commandBuffers[i],static_cast<uint32_t>(proxyCube.indices.size()),1,0,0,0);
        }
        {
            vkCmdNextSubpass(commandBuffers[i],VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,graphicsPipelines.rayCast);

            vkCmdBindDescriptorSets(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayouts.rayCast,
                                    0,1,&descriptorSets.rayCast[i],0,nullptr);

            vkCmdDraw(commandBuffers[i],6,1,0,0);
        }
        vkCmdEndRenderPass(commandBuffers[i]);

        if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS){
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}
void VolumeViewerApplication::createSyncObjects()
{
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(),VK_NULL_HANDLE);

    VkSemaphoreCreateInfo  semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for(size_t i =0;i<MAX_FRAMES_IN_FLIGHT;i++){
        if(vkCreateSemaphore(device,&semaphoreInfo, nullptr,&imageAvailableSemaphores[i])!=VK_SUCCESS
           || vkCreateSemaphore(device,&semaphoreInfo,nullptr,&renderFinishedSemaphores[i])!=VK_SUCCESS
           || vkCreateFence(device,&fenceInfo,nullptr,&inFlightFences[i])!=VK_SUCCESS){
            throw std::runtime_error("failed to create synchronization objects for a frame");
        }
    }
}
void VolumeViewerApplication::drawFrame()
{
    vkWaitForFences(device,1,&inFlightFences[currentFrame],VK_TRUE,UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device,swapChain,UINT64_MAX,imageAvailableSemaphores[currentFrame],
                                            VK_NULL_HANDLE,&imageIndex);

    if(result == VK_ERROR_OUT_OF_DATE_KHR){
        return;
    }
    else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR){
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    updateUniformBuffer(imageIndex);

    if(imagesInFlight[imageIndex] != VK_NULL_HANDLE){
        vkWaitForFences(device,1,&imagesInFlight[imageIndex],VK_TRUE,UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vkResetFences(device,1,&inFlightFences[currentFrame]);

    if(vkQueueSubmit(graphicsQueue,1,&submitInfo,inFlightFences[currentFrame])!=VK_SUCCESS){
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    VkSwapchainKHR swapChains[] = {swapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue,&presentInfo);

    if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR){

    }
    else if(result != VK_SUCCESS){
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}
void VolumeViewerApplication::updateUniformBuffer(uint32_t currentImage)
{
    //mvp
    {
        MVP mvp_ubo;
        mvp_ubo.model = glm::mat4(1.f);
        mvp_ubo.view = camera->getViewMatrix();
        mvp_ubo.proj = glm::perspective(glm::radians(45.f),swapChainExtent.width/(float)swapChainExtent.height,
                                        0.01f,8.f);
        mvp_ubo.proj[1][1] *= -1.f;

        void* data;
        vkMapMemory(device,mvpUBO[currentImage].bufferMemory,0,sizeof(MVP),0,&data);
        memcpy(data,&mvp_ubo,sizeof(MVP));
        vkUnmapMemory(device,mvpUBO[currentImage].bufferMemory);
    }
    //view pos
    {
        glm::vec4 viewPos_ubo;
        viewPos_ubo = glm::vec4(1.28f,1.28f,4.6f,0.f);

        void* data;
        vkMapMemory(device,viewPosUBO[currentImage].bufferMemory,0,sizeof(glm::vec4),0,&data);
        memcpy(data,&viewPos_ubo,sizeof(glm::vec4));
        vkUnmapMemory(device,viewPosUBO[currentImage].bufferMemory);
    }
    static size_t cnt = 0;
    //volume info
    if(cnt<3){
        void* data;
        vkMapMemory(device,volumeInfoUBO[currentImage].bufferMemory,0,sizeof(VolumeInfo),0,&data);
        memcpy(data,&volumeInfo,sizeof(VolumeInfo));
        vkUnmapMemory(device,volumeInfoUBO[currentImage].bufferMemory);
    }
    cnt++;
}

int main(int argc,char** argv){

    try{
        VolumeViewerApplication app;
        app.run();
    }
    catch (const std::exception& err)
    {
        LOG_ERROR("program exit with error: {}",err.what());
    }

    return 0;
}