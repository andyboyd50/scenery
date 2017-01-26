#version 450 core
#extension GL_ARB_separate_shader_objects: enable

#define INVROOT2 .7071067

layout(location = 0) in VertexData {
    vec3 Position;
    vec3 Normal;
    vec2 TexCoord;
    vec3 FragPosition;
} VertexIn;

layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec4 gAlbedoSpec;

const float PI = 3.14159265358979323846264;
const int NUM_OBJECT_TEXTURES = 5;

struct MaterialInfo {
    vec3 Ka;
    vec3 Kd;
    vec3 Ks;
    float Shininess;
};

const int MATERIAL_HAS_DIFFUSE =  0x0001;
const int MATERIAL_HAS_AMBIENT =  0x0002;
const int MATERIAL_HAS_SPECULAR = 0x0004;
const int MATERIAL_HAS_NORMAL =   0x0008;

layout(binding = 0) uniform Matrices {
	mat4 ViewMatrix;
	mat4 ModelMatrix;
	mat4 ProjectionMatrix;
	mat4 MVP;
	vec3 CamPosition;
	int isBillboard;
} ubo;

layout(binding = 1) uniform MaterialProperties {
    MaterialInfo Material;
    int materialType;
};

/*
    ObjectTextures[0] - ambient
    ObjectTextures[1] - diffuse
    ObjectTextures[2] - specular
    ObjectTextures[3] - normal
    ObjectTextures[4] - displacement
*/

layout(set = 1, binding = 0) uniform sampler2D ObjectTextures[NUM_OBJECT_TEXTURES];

float square(float x) {
    return x*x;
}

float squaredMagnitude(vec2 uv) {
    return dot(uv, uv);
}

float signNotZero(float k) {
    return k >= 0.0 ? 1.0 : -1.0;
}

vec2 encodeSphericalNormals(vec3 v) {
    float thetaNorm = acos(v.y) / PI;
    float phiNorm = (atan(v.x, v.z) / PI) * 0.5 + 0.5;

    return vec2(phiNorm, thetaNorm);
}

vec2 encodeSpheremapNormals(vec3 normal) {
//  Spheremap, Lambert Azimuthal Equal-Area projection
//    float p = sqrt(8.0 * normal.z + 8.0);
//    return normal.xy/p + 0.5;
//  CryEngine3
    vec2 encoded = normalize(normal.xy) * (sqrt(normal.z * 0.5 + 0.5));
//
//    return encoded;
//  Stereographic projection
//    float zsign = signNotZero(normal.z);
//    float scale = 1.7777;
//    vec2 encoded = normal.xy/(abs(normal.z)+1.0);
//    encoded /= scale;
//
//    encoded = encoded*0.5+0.5;
    return encoded;
}

void ellipseFromDisk(vec2 uv) {
    /**Convert from the disk to the square*/
    float uSqMinusVSq = square(uv.x) - square(uv.y);
    // making this constant 2.0f causes a small set of vectors to be mapped off by 90 degrees
    // because of floating point rounding errors. When set to 1.9999995f this error is no longer
    // an issue
    float t1 = 1.9999995 + uSqMinusVSq;
    //float t1 = 2.0f + uSqMinusVSq;
    float t2 = sqrt(-8.0f * square(uv.x) + t1*t1);
    float newU = sqrt(2.0f + uSqMinusVSq - t2) * INVROOT2 * sign(uv.x);
    float newV = 2.0f*uv.y * inversesqrt(2.0f - uSqMinusVSq + t2);
    uv = vec2(newU, newV);
}

vec2 encodeEANormals(vec3 vec) {
    float denominator = inversesqrt(abs(vec.z) + 1.0f);
    vec2 uv = vec.xy * denominator;
    ellipseFromDisk(uv);
    /**Pack sign of z into sign of u*/
    float zsign = signNotZero(vec.z);
    uv.x = ((uv.x * 0.5f) + 0.5f) * zsign;
    return uv;
}

mat3 TBN(vec3 N, vec3 position, vec2 uv) {
    vec3 dp1 = dFdx(position);
    vec3 dp2 = dFdy(position);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);

    vec3 dp2Perpendicular = cross(dp2, N);
    vec3 dp1Perpendicular = cross(N, dp1);

    vec3 T = dp2Perpendicular * duv1.x + dp1Perpendicular * duv2.x;
    vec3 B = dp2Perpendicular * duv1.y + dp1Perpendicular * duv2.y;

    float invmax = inversesqrt(max(dot(T, T), dot(B, B)));

    return mat3(T * invmax, B * invmax, N);
}

void main() {
    // Store the fragment position vector in the first gbuffer texture
    gPosition = VertexIn.FragPosition;
    gAlbedoSpec.rgb = vec3(0.0f, 0.0f, 0.0f);

    gAlbedoSpec.rgb = Material.Kd;
    gAlbedoSpec.a = Material.Ka.r*Material.Shininess;

    vec3 V = mat3(ubo.ViewMatrix*ubo.ModelMatrix)*(VertexIn.FragPosition - ubo.CamPosition);
    mat3 tbn = TBN(normalize(VertexIn.Normal), V, VertexIn.TexCoord);

    // Also store the per-fragment normals into the gbuffer
    if((materialType & MATERIAL_HAS_AMBIENT) == MATERIAL_HAS_AMBIENT) {
        //gAlbedoSpec.rgb = texture(ObjectTextures[0], VertexIn.TexCoord).rgb;
    }

    if((materialType & MATERIAL_HAS_DIFFUSE) == MATERIAL_HAS_DIFFUSE) {
        gAlbedoSpec.rgb = texture(ObjectTextures[1], VertexIn.TexCoord).rgb;
    }

    if((materialType & MATERIAL_HAS_SPECULAR) == MATERIAL_HAS_SPECULAR) {
        gAlbedoSpec.a = log2(texture(ObjectTextures[2], VertexIn.TexCoord).r)/10.5;
    }

    if((materialType & MATERIAL_HAS_NORMAL) == MATERIAL_HAS_NORMAL) {
        vec3 normal = normalize(texture(ObjectTextures[3], VertexIn.TexCoord).rgb*255.0/127.0 - 128.0/127.0);

        gNormal = vec3(normalize(normal));
    } else {
        gNormal = vec3(normalize(VertexIn.Normal));
    }
}
