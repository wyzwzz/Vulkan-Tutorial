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

#include <json.hpp>

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
#include <spdlog/stopwatch.h>

const std::string assetPath = "C:/Users/wyz/projects/Vulkan-Tutorial/data/";

const std::string modelPath = "E:/neurons/neurons.json";

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

struct Mesh{
    struct Surface{
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::string name;
        glm::vec4 color;
    };
    std::vector<Surface> surfaces;
    Mesh() = default;
    ~Mesh() = default;
    void Load(const std::string& filename){
        auto ext = filename.substr(filename.find_last_of('.'));
        if(ext == ".obj"){
            readNeuron({"default",filename,{1.f,1.f,1.f,1.f}});
        }
        else if(ext == ".json"){
            auto neurons = load(filename);
            for(const auto& neuron:neurons){
                spdlog::stopwatch sw;
                readNeuron(neuron);
                LOG_INFO("load {0} cost time {1}",neuron.path,sw);
            }
        }
        else{
            throw std::runtime_error("Mesh Load with invalid file format!");
        }
        LOG_INFO("mesh load surface num: {}",surfaces.size());
    }
    void Transform(const glm::vec3& trans){
        for(auto& surface:surfaces){
            for(auto& vertex:surface.vertices){
                vertex.pos *= trans;
            }
        }
    }
  private:
    struct Neuron{
        std::string name;
        std::string path;
        glm::vec4 color;
    };
    std::vector<Neuron> load(const std::string& filename){
        std::ifstream in(filename);
        if (!in.is_open())
        {
            throw std::runtime_error("mesh config file open failed");
        }
        nlohmann::json j;
        in >> j;
        if (j.find("neurons") == j.end())
        {
            throw std::runtime_error("wrong config file format");
        }
        auto neurons = j.at("neurons");
        if (neurons.find("count") == neurons.end())
        {
            throw std::runtime_error("invalid config file format");
        }
        int count = neurons.at("count");
        auto resource = neurons.at("resource");
        if (count != resource.size())
        {
            LOG_ERROR("count {0} not equal to resource size {1}", count, resource.size());
        }
        bool is_same_color = neurons.at("neuron_color").at("use_same_color") == "true";
        std::array<float, 4> same_color = neurons.at("neuron_color").at("color");

        std::vector<Neuron> read_neurons;
        int cnt = 0;
        for (auto &res : resource)
        {
            std::string name, path;
            std::array<float, 4> color_v = {1.f, 1.f, 1.f, 1.f};
            if (res.find("name") == res.end())
            {
                name = "default" + std::to_string(cnt);
                cnt++;
            }
            else
            {
                name = res.at("name");
            }
            if (res.find("path") == res.end())
            {
                LOG_ERROR("path not find, config file error");
                continue;
            }
            path = res.at("path");
            if (res.find("color") == res.end())
            {
                if (is_same_color)
                {
                    color_v = same_color;
                }
                else
                {
                    LOG_INFO("Not found color, use default color white");
                }
            }
            else
            {
                if (is_same_color)
                {
                    color_v = same_color;
                }
                else
                {
                    std::vector<float> color = res.at("color");
                    std::copy(color.begin(), color.end(), color_v.begin());
                }
            }
            read_neurons.emplace_back(
                Neuron{name, path,glm::vec4{color_v[0],color_v[1],color_v[2],color_v[3]}}
                );
        }
        return read_neurons;
    }
    void readNeuron(const Neuron& neuron){
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn,err;
        if(!tinyobj::LoadObj(&attrib,&shapes,&materials,&warn,&err,neuron.path.c_str())){
            throw std::runtime_error(warn+err);
        }

        std::unordered_map<Vertex,uint32_t> uniqueVertices{};

        this->surfaces.emplace_back();
        auto& surface = surfaces.back();
        surface.name = neuron.name;
        surface.color = neuron.color;
        auto& vertices = surface.vertices;
        auto& indices = surface.indices;
        for(const auto& shape:shapes){
            for(const auto& index:shape.mesh.indices){
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2]
                };

                if(uniqueVertices.count(vertex) == 0){
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }
};


class MeshViewerApplication{
    GLFWwindow* window;
    int window_w = 1920;
    int window_h = 1080;
    int iGPU = 0;

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

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkCommandPool commandPool;

    VkImage colorImage;//for multi sample render target
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    std::vector<VkFramebuffer> swapChainFramebuffers;

    //---------------------------------------------------------------------------

    struct DrawModel{
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexbufferMemory;
        glm::vec4 color;
        std::string name;
    };
    std::vector<DrawModel> drawModels;

    struct UBO{
        VkBuffer buffer;
        VkDeviceMemory bufferMemory;
    };

    struct MVP{
        alignas(16) glm::mat4 model;
        alignas(16) glm::mat4 view;
        alignas(16) glm::mat4 proj;
    };
    std::vector<UBO> mvp;
    struct Shading{
        struct {
            alignas(16) glm::vec4 position;
            alignas(16) glm::vec4 color;
        } light;
        alignas(16) glm::vec4 viewPos;
    };
    std::vector<UBO> shading;

    VkDescriptorPool descriptorPool;

    std::vector<VkDescriptorSet> descriptorSets;//for each swap chain image in order to multi-thread draw

    std::vector<VkCommandBuffer> commandBuffers;

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
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);

        window = glfwCreateWindow(window_w,window_h,"MPIMeshViewer",nullptr,nullptr);

        glfwSetWindowUserPointer(window,this);
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
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createColorResources();
        createDepthResources();
        createFramebuffers();
        //3.draw dynamic vulkan resources
        loadModel();
        createModelResources();
        createUniformBuffers();
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

        vkDestroyImageView(device,colorImageView,nullptr);
        vkDestroyImage(device,colorImage,nullptr);
        vkFreeMemory(device,colorImageMemory,nullptr);

        vkDestroyImageView(device,depthImageView,nullptr);
        vkDestroyImage(device,depthImage,nullptr);
        vkFreeMemory(device,depthImageMemory,nullptr);

        for(auto framebuffer : swapChainFramebuffers){
            vkDestroyFramebuffer(device,framebuffer,nullptr);
        }

        vkFreeCommandBuffers(device,commandPool,static_cast<uint32_t>(commandBuffers.size()),commandBuffers.data());

        vkDestroyPipeline(device,graphicsPipeline,nullptr);
        vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
        vkDestroyRenderPass(device,renderPass,nullptr);

        vkDestroySwapchainKHR(device,swapChain,nullptr);

        vkDestroyDescriptorSetLayout(device,descriptorSetLayout,nullptr);

        for(auto& model:drawModels){
            vkDestroyBuffer(device,model.vertexBuffer,nullptr);
            vkFreeMemory(device,model.vertexBufferMemory,nullptr);
            vkDestroyBuffer(device,model.indexBuffer,nullptr);
            vkFreeMemory(device,model.indexbufferMemory,nullptr);
        }

        vkDestroyCommandPool(device,commandPool,nullptr);

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
    //create vulkan default resources
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createSwapChainImageViews();
    //setup graphics pipeline
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    //setup graphics resources
    void createCommandPool();
    //1.framebuffer resources
    void createColorResources();//for multi sample render target
    void createDepthResources();
    void createFramebuffers();
    //2.model resources
    void loadModel();
    void createModelResources();//vertex and index
    //3.shader descriptor set resources
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    //setup draw commands and present sync resources
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
    VkFormat findDepthFormat();
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,VkImageTiling tiling,VkFormatFeatureFlags features);
    static std::vector<char> readFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createImage(uint32_t width,uint32_t height,uint32_t mipLevels,
                     VkSampleCountFlagBits numSamples,VkFormat format,
                     VkImageTiling tiling,VkImageUsageFlags usage,VkMemoryPropertyFlags properties,
                     VkImage& image,VkDeviceMemory& imageMemory);
    uint32_t findMemoryType(uint32_t typeFilter,VkMemoryPropertyFlags properties);
    void createVertexBuffer(VkBuffer& vertexBuffer,VkDeviceMemory& vertexBufferMemory,const std::vector<Vertex>& vertices);
    void createIndexBuffer(VkBuffer& indexBuffer,VkDeviceMemory& indexBufferMemory,const std::vector<uint32_t>& indices);
    void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags properties,
                      VkBuffer& buffer,VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer,VkBuffer dstBuffer,VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void updateUniformBuffer(uint32_t currentImage);
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
void MeshViewerApplication::createRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;//clear preserve value
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;//don't care previous layout
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;//layout during the subpass

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
    subpass.inputAttachmentCount = 0;//use color attachment as input
    subpass.pInputAttachments = nullptr;

    // Subpass dependencies for layout transitions
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription,3> attachments =
        {colorAttachment,depthAttachment,colorAttachmentResolve};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if(vkCreateRenderPass(device,&renderPassInfo, nullptr,&renderPass)!=VK_SUCCESS){
        throw std::runtime_error("failed to create render pass!");
    }

}
void MeshViewerApplication::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding mvpLayoutBinding{};
    mvpLayoutBinding.binding = 0;
    mvpLayoutBinding.descriptorCount = 1;
    mvpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    mvpLayoutBinding.pImmutableSamplers = nullptr;
    mvpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 1;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding,2> bindings = {mvpLayoutBinding,uboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if(vkCreateDescriptorSetLayout(device,&layoutInfo, nullptr,&descriptorSetLayout)!=VK_SUCCESS){
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}
void MeshViewerApplication::createGraphicsPipeline()
{
    auto vertShaderCode = readFile(assetPath+"shaders/MeshViewer/shader.vert.spv");
    auto fragShaderCode = readFile(assetPath+"shaders/MeshViewer/shader.frag.spv");

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
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;

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

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.f;
    colorBlending.blendConstants[1] = 0.f;
    colorBlending.blendConstants[2] = 0.f;
    colorBlending.blendConstants[3] = 0.f;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    if(vkCreatePipelineLayout(device,&pipelineLayoutInfo,nullptr,&pipelineLayout)!=VK_SUCCESS){
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
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if(vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&pipelineInfo, nullptr,&graphicsPipeline)!=VK_SUCCESS){
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(device,fragShaderModule, nullptr);
    vkDestroyShaderModule(device,vertShaderModule, nullptr);
}
VkFormat MeshViewerApplication::findDepthFormat()
{
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}
VkFormat MeshViewerApplication::findSupportedFormat(const std::vector<VkFormat> &candidates, VkImageTiling tiling,
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
std::vector<char> MeshViewerApplication::readFile(const std::string &filename)
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
VkShaderModule MeshViewerApplication::createShaderModule(const std::vector<char> &code)
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
void MeshViewerApplication::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if(vkCreateCommandPool(device,&poolInfo,nullptr,&commandPool)!=VK_SUCCESS){
        throw std::runtime_error("failed to create command pool!");
    }
}
void MeshViewerApplication::createColorResources()
{
    VkFormat colorFormat = swapChainImageFormat;

    //multi sample image must with mipLevels 1
    createImage(swapChainExtent.width,swapChainExtent.height,
                1,msaaSamples,
                colorFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                colorImage,colorImageMemory);
    colorImageView = createImageView(colorImage,colorFormat,
                                     VK_IMAGE_ASPECT_COLOR_BIT,1);
}
void MeshViewerApplication::createDepthResources()
{
    VkFormat depthFormat = findDepthFormat();
    createImage(swapChainExtent.width,swapChainExtent.height,
                1,msaaSamples,
                depthFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                depthImage,depthImageMemory);
    depthImageView = createImageView(depthImage,depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT,1);
}
void MeshViewerApplication::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());

    for(size_t i =0;i<swapChainImageViews.size();i++){
        //attachments should compatible with render pass setting
        //here are attachments instances so should compatible with render pass which already set attachments type info
        std::array<VkImageView,3> attachments = {
            colorImageView,
            depthImageView,
            swapChainImageViews[i]
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
void MeshViewerApplication::createImage(uint32_t width, uint32_t height,
                                        uint32_t mipLevels,VkSampleCountFlagBits numSamples,
                                        VkFormat format, VkImageTiling tiling,VkImageUsageFlags usage,
                                        VkMemoryPropertyFlags properties,
                                        VkImage &image,VkDeviceMemory &imageMemory)
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
uint32_t MeshViewerApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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
void MeshViewerApplication::loadModel()
{
    Mesh mesh;
    mesh.Load(modelPath);
    mesh.Transform({0.00032f,0.00032f,0.00032f});
    drawModels.resize(mesh.surfaces.size());
    for(int i=0;i<mesh.surfaces.size();i++){
        auto& s = mesh.surfaces[i];
        drawModels[i].name = s.name;
        drawModels[i].color = s.color;
        drawModels[i].vertices = std::move(s.vertices);
        drawModels[i].indices = std::move(s.indices);
    }
    if(drawModels.empty()){
        LOG_ERROR("load model is empty");
    }
}
void MeshViewerApplication::createModelResources()
{
    LOG_INFO("{}",__FUNCTION__ );
    for(auto& model:drawModels){
        createVertexBuffer(model.vertexBuffer,model.vertexBufferMemory,model.vertices);
        createIndexBuffer(model.indexBuffer,model.indexbufferMemory,model.indices);
    }
}
void MeshViewerApplication::createVertexBuffer(VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory,
                                               const std::vector<Vertex>& vertices)
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
void MeshViewerApplication::createIndexBuffer(VkBuffer &indexBuffer, VkDeviceMemory &indexBufferMemory,
                                              const std::vector<uint32_t>& indices)
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
void MeshViewerApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                         VkBuffer &buffer, VkDeviceMemory &bufferMemory)
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
void MeshViewerApplication::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer,srcBuffer,dstBuffer,1,&copyRegion);

    endSingleTimeCommands(commandBuffer);
}
VkCommandBuffer MeshViewerApplication::beginSingleTimeCommands()
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
void MeshViewerApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer)
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
void MeshViewerApplication::createUniformBuffers()
{
    LOG_INFO("{}",__FUNCTION__ );
    //mvp
    {
        VkDeviceSize bufferSize = sizeof(MVP);

        mvp.resize(swapChainImages.size());
        for(size_t i=0;i<swapChainImages.size();i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         mvp[i].buffer,mvp[i].bufferMemory);
        }
    }
    //shading
    {
        VkDeviceSize bufferSize = sizeof(Shading);

        shading.resize(swapChainImages.size());
        for(size_t i=0;i<swapChainImages.size();i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         shading[i].buffer,shading[i].bufferMemory);
        }
    }
}
void MeshViewerApplication::createDescriptorPool()
{
    LOG_INFO("{}",__FUNCTION__ );
    std::array<VkDescriptorPoolSize,2> poolSize{};
    poolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size()+2);
    poolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size()+2);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

    if(vkCreateDescriptorPool(device,&poolInfo, nullptr,&descriptorPool)!=VK_SUCCESS){
        throw std::runtime_error("failed to create descriptor pool!");
    }
}
void MeshViewerApplication::createDescriptorSets()
{
    LOG_INFO("{}",__FUNCTION__ );
    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(),descriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.resize(swapChainImages.size());
    if(vkAllocateDescriptorSets(device,&allocInfo,descriptorSets.data())!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for(size_t i =0;i<swapChainImages.size();i++){
        VkDescriptorBufferInfo mvpBufferInfo{};
        mvpBufferInfo.buffer = mvp[i].buffer;
        mvpBufferInfo.offset = 0;
        mvpBufferInfo.range = sizeof(MVP);

        VkDescriptorBufferInfo shadingBufferInfo{};
        shadingBufferInfo.buffer = shading[i].buffer;
        shadingBufferInfo.offset = 0;
        shadingBufferInfo.range = sizeof(Shading);

        std::array<VkWriteDescriptorSet,2> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &mvpBufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &shadingBufferInfo;

        vkUpdateDescriptorSets(device,
                               static_cast<uint32_t>(descriptorWrites.size()),
                               descriptorWrites.data(),0,nullptr);
    }
}
void MeshViewerApplication::createCommandBuffers()
{
    LOG_INFO("{}",__FUNCTION__ );
    commandBuffers.resize(swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = (uint32_t) commandBuffers.size();
    if(vkAllocateCommandBuffers(device,&allocateInfo,commandBuffers.data())!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate command buffers!");
    }

    for(size_t i =0;i<commandBuffers.size();i++){
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

        std::array<VkClearValue,2> clearValues;
        clearValues[0].color = {0.f,0.f,0.f,1.f};
        clearValues[1].depthStencil = {1.0f,0};

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        //begin a render pass
        vkCmdBeginRenderPass(commandBuffers[i],&renderPassInfo,VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,graphicsPipeline);

        for(auto& model:drawModels){
            VkDeviceSize offsets[]={0};
            vkCmdBindVertexBuffers(commandBuffers[i],0,1,&model.vertexBuffer,offsets);

            vkCmdBindIndexBuffer(commandBuffers[i],model.indexBuffer,0,VK_INDEX_TYPE_UINT32);

            vkCmdBindDescriptorSets(commandBuffers[i],VK_PIPELINE_BIND_POINT_GRAPHICS,pipelineLayout,
                                    0,1,&descriptorSets[i],0,nullptr);
            //todo should use push constant
            vkCmdDrawIndexed(commandBuffers[i],static_cast<uint32_t>(model.indices.size()),1,0,0,0);

        }

        //end a render pass
        vkCmdEndRenderPass(commandBuffers[i]);

        if(vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS){
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}
void MeshViewerApplication::createSyncObjects()
{
    LOG_INFO("{}",__FUNCTION__ );
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
void MeshViewerApplication::drawFrame()
{
    LOG_INFO("{}",__FUNCTION__ );
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
void MeshViewerApplication::updateUniformBuffer(uint32_t currentImage)
{
    {
        MVP mvp_ubo{};
        mvp_ubo.model = glm::mat4(1.f);
        mvp_ubo.view = glm::lookAt(glm::vec3(1.f, 1.f, 2.f), glm::vec3(1.f, 1.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
        mvp_ubo.proj =
            glm::perspective(glm::radians(45.f), swapChainExtent.width / (float)swapChainExtent.height, 0.01f, 5.f);
        mvp_ubo.proj[1][1] *= -1.f;

        void *data;
        vkMapMemory(device, mvp[currentImage].bufferMemory, 0, sizeof(MVP), 0, &data);
        memcpy(data, &mvp_ubo, sizeof(MVP));
        vkUnmapMemory(device, mvp[currentImage].bufferMemory);
    }
    {
        Shading shading_ubo{};
        shading_ubo.viewPos = glm::vec4{1.f, 1.f, 2.f, 1.f};
        shading_ubo.light = {glm::vec4{3.f, 3.f, 3.f, 1.f}, {1.f, 0.8f, 0.6f, 1.f}};

        void* data;
        vkMapMemory(device,shading[currentImage].bufferMemory,0,sizeof(Shading),0,&data);
        memcpy(data,&shading_ubo,sizeof(Shading));
        vkUnmapMemory(device,shading[currentImage].bufferMemory);
    }
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