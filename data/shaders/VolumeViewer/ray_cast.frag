#version 450

layout(location = 0) out vec4 outColor;

layout(input_attachment_index = 0,binding = 0) uniform subpassInput rayEntryPos;

layout(input_attachment_index = 1,binding = 1) uniform subpassInput rayExitPos;

layout(binding = 2) uniform sampler1D transferFunc;

layout(binding = 3) uniform sampler3D volumeData;

layout(binding = 4) uniform VolumeInfoUBO{
    float volume_board_x;
    float volume_board_y;
    float volume_board_z;
    float step;
} volumeInfoUBO;

void main() {
    vec3 ray_entry_pos = subpassLoad(rayEntryPos).xyz;
    vec3 ray_exit_pos = subpassLoad(rayExitPos).xyz;
    vec3 entry_to_exit = ray_exit_pos - ray_entry_pos;
    vec3 ray_direction = normalize(entry_to_exit);
//    outColor = vec4(ray_direction,1.f);
//    return ;
    float distance = dot(ray_direction,entry_to_exit);
    int steps = int(distance / volumeInfoUBO.step);
    vec3 ray_sample_pos = ray_entry_pos;
    vec4 color = vec4(0.f);
    vec3 volume_board = vec3(volumeInfoUBO.volume_board_x,volumeInfoUBO.volume_board_y,volumeInfoUBO.volume_board_z);
    for(int i = 0; i < steps; i++){
        vec3 ray_sample_tex_pos = ray_sample_pos / volume_board;
        float scalar = texture(volumeData,ray_sample_tex_pos).r;
        vec4 sample_color = texture(transferFunc,scalar);
        color = color + sample_color * vec4(sample_color.aaa,1.f) * (1.f - color.a);
        if(color.a > 0.99f){
            break;
        }
        ray_sample_pos = ray_entry_pos + ray_direction * (i+1) * volumeInfoUBO.step;
    }
//    if(color.a == 0.f)
//        discard;
    color.a = 1.f;
    outColor = color;
}
