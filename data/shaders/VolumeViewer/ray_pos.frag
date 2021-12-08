#version 450

layout(location = 0) in vec3 inFragPos;

layout(location = 0) out vec4 entryPos;

layout(location = 1) out vec4 exitPos;

layout(binding = 1) uniform ViewPos{
    vec4 view_pos;
}viewPos;

void main() {
    bool inner = viewPos.view_pos.w == 1.f;
    if(gl_FrontFacing){
        if(inner){
            entryPos = viewPos.view_pos;
        }
        else{
            entryPos = vec4(inFragPos,1.f);
        }
        exitPos = vec4(0.f);
    }
    else{
        entryPos = vec4(0.f);
        exitPos = vec4(inFragPos,1.f);
    }
}
