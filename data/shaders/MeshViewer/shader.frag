#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

struct Light{
    vec4 position;
    vec4 color;
};

layout(binding = 1) uniform UBO{
    Light light;
    vec4 view_pos;
}ubo;

void main() {
    vec3 ambient = 0.05f * ubo.light.color.rgb;

    vec3 normal = normalize(inNormal);
    vec3 light_dir = normalize(ubo.light.position.xyz-inPosition);
    float diff = max(dot(light_dir,normal),0.f);
    vec3 diffuse = diff * ubo.light.color.rgb;

    vec3 view_dir = normalize(ubo.view_pos.xyz - inPosition);
    vec3 halfway_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal,halfway_dir),0.f),32.f);
    vec3 specular = vec3(0.3f)*spec;
    outColor = vec4(ambient + diffuse + specular, 1.f);
}
