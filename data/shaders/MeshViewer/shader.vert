#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;

layout(binding = 0) uniform MVPMatrix {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model* vec4(inPosition, 1.0);
    outPosition = vec3(mvp.model * vec4(inPosition,1.f));
    outNormal = vec3(mvp.model * vec4(inNormal,0.f));
}
