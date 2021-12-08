#version 450

layout(location = 0) in vec3 inVertexPos;

layout(location = 0) out vec3 outVertexPos;

layout(binding = 0) uniform MVPMatrix{
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(inVertexPos, 1.0);
    outVertexPos = vec3(mvp.model * vec4(inVertexPos,1.f));
}
