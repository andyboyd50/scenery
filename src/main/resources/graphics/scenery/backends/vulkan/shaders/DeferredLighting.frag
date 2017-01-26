#version 450 core
#extension GL_ARB_separate_shader_objects: enable

#define PI 3.14159265359
#define INVROOT2 .7071067

layout(location = 0) in vec2 textureCoord;
layout(location = 0) out vec4 FragColor;

struct Light {
	float Linear;
	float Quadratic;
	float Intensity;
	vec4 Position;
  	vec4 Color;
};

const int MAX_NUM_LIGHTS = 128;

layout(binding = 0) uniform Matrices {
	mat4 ViewMatrix;
	mat4 ModelMatrix;
	mat4 ProjectionMatrix;
	mat4 MVP;
	vec3 CamPosition;
	int isBillboard;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D gPosition;
layout(set = 1, binding = 1) uniform sampler2D gNormal;
layout(set = 1, binding = 2) uniform sampler2D gAlbedoSpec;
layout(set = 1, binding = 3) uniform sampler2D gDepth;

layout(set = 2, binding = 0, std140) uniform LightParameters {
    int numLights;
	Light lights[MAX_NUM_LIGHTS];
};

layout(set = 3, binding = 0, std140) uniform ShaderParameters {
	int debugBuffers;
	int activateSSAO;
	int reflectanceModel;
	float ssaoDistanceThreshold;
	vec2 ssaoFilterRadius;
};

const vec2 poisson16[] = vec2[](    // These are the Poisson Disk Samples
		vec2( -0.94201624,  -0.39906216 ),
		vec2(  0.94558609,  -0.76890725 ),
		vec2( -0.094184101, -0.92938870 ),
		vec2(  0.34495938,   0.29387760 ),
		vec2( -0.91588581,   0.45771432 ),
		vec2( -0.81544232,  -0.87912464 ),
		vec2( -0.38277543,   0.27676845 ),
		vec2(  0.97484398,   0.75648379 ),
		vec2(  0.44323325,  -0.97511554 ),
		vec2(  0.53742981,  -0.47373420 ),
		vec2( -0.26496911,  -0.41893023 ),
		vec2(  0.79197514,   0.19090188 ),
		vec2( -0.24188840,   0.99706507 ),
		vec2( -0.81409955,   0.91437590 ),
		vec2(  0.19984126,   0.78641367 ),
		vec2(  0.14383161,  -0.14100790 )
);

vec3 decodeSphericalNormals(vec2 encoded) {
    float theta = encoded.y * PI;
    float phi = (encoded.x * 2.0 * PI - PI);

    float sintheta = sin(theta);
    return vec3(sintheta * sin(phi), cos(theta), sintheta*cos(phi));
}

vec3 decodeSpheremapNormals(vec2 encoded) {
//  Lambert Azimuthal Equal-Area projection
//    vec2 fEncoded = encoded * 4.0 - 2.0;
//    float f = dot(fEncoded, fEncoded);
//    float g = sqrt(1.0-f/4.0);
//
//    vec3 n;
//    n.xy = fEncoded * g;
//    n.z = 1.0 - f/2.0;
//
//    return n;
//  CryEngine3
//    vec4 nn = vec4(encoded, 0.0, 0.0) * vec4(2.0, 2.0, 0.0, 0.0) + vec4(-1, -1, 1, -1);
    float l = dot(encoded, encoded);
    vec3 normal;
    normal.z = l*2-1;
    normal.xy = normalize(encoded.xy)*sqrt(1-normal.z*normal.z);
//
//    return nn.xyz * 2.0 + vec3(0, 0, -1);
//  Stereographic
//    float scale = 1.7777;
//
//    vec3 nn = vec3(encoded, 0.0) * vec3(2.0*scale, 2.0*scale, 0.0) + vec3(-scale, -scale, 1.0);
//    float g = 2.0 / dot(nn.xyz, nn.xyz);
//
//    vec3 normal = vec3(0.0);
//    normal.xy = g * nn.xy;
//    normal.z = (g - 1.0);

    return normal;
}

float square(float x) {
    return x*x;
}

void ellipseToDisk(inout vec2 uv) {
    float x = uv.x * sqrt(1.0 - square(uv.y) * 0.5);
    float y = uv.y * sqrt(1.0 - square(uv.x) * 0.5);
    uv = vec2(x, y);
}

void ellipseFromDisk(inout vec2 uv) {
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

float squaredMagnitude(vec2 uv) {
    return dot(uv, uv);
}

vec3 decodeEANormals(vec2 encoded) {
    float zsign   = sign(encoded.x);
    float u       = (abs(encoded.x) - .5) * 2.0;
    vec2 uv = vec2(encoded.x, encoded.y);
    /**map back from the square to the disk*/
    ellipseToDisk(uv);

    /**return the vector using the map*/
    float r2 = squaredMagnitude(uv);
    float temp = sqrt((1.0 - square(1.0 - r2)) / r2);
    return vec3 ( uv.x * temp, uv.y * temp, zsign * (1.0f - r2) );
}

vec3 calculatePosition(vec2 texCoord, float depth) {
	vec4 pos = inverse(ubo.ProjectionMatrix) * vec4(texCoord.x * 2 - 1, texCoord.y * 2 - 1, depth * 2 - 1, 1);
	return pos.xyz;
}

float GGXDistribution(vec3 normal, vec3 halfway, float roughness) {
    float a = roughness*roughness;
    float aSquared = a*a;
    float NdotH = max(dot(normal, halfway), 0.0);
    float NdotH2 = NdotH*NdotH;

    float denom = ((NdotH2 * (aSquared - 1.0) + 1.0));
    return aSquared/(denom*denom*PI);
}

float GeometrySchlick(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    return (NdotV)/(NdotV*(1.0 - k) + k);
}

float GeometrySmith(vec3 normal, vec3 view, vec3 light, float roughness) {
    float NdotV = max(dot(normal, view), 0.0);
    float NdotL = max(dot(normal, light), 0.0);

    return GeometrySchlick(NdotV, roughness) * GeometrySchlick(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
	// Retrieve data from G-buffer
	vec3 FragPos = texture(gPosition, textureCoord).rgb;
	vec3 Normal = normalize(decodeSpheremapNormals(texture(gNormal, textureCoord).rg));
//    vec3 Normal = texture(gNormal, textureCoord).rgb;
	vec4 Albedo = texture(gAlbedoSpec, textureCoord).rgba;
	float Specular = exp(texture(gAlbedoSpec, textureCoord).a)*10.5;
	float Depth = texture(gDepth, textureCoord).r;

	float fragDist = length(FragPos - ubo.CamPosition);

	vec3 lighting = vec3(0.0);

	if(debugBuffers == 0) {
		float ambientOcclusion = 0.0f;

		if(activateSSAO > 0) {

			int sample_count = 8;
			for (int i = 0; i < sample_count;  ++i) {
				// sample at an offset specified by the current Poisson-Disk sample and scale it by a radius (has to be in Texture-Space)
				vec2 sampleTexCoord = textureCoord + (poisson16[i] * (ssaoFilterRadius));
				float sampleDepth = texture(gDepth, sampleTexCoord).r;
				vec3 samplePos = texture(gPosition, sampleTexCoord).rgb;//calculatePosition(sampleTexCoord, sampleDepth);

				vec3 sampleDir = normalize(samplePos - FragPos);

				// angle between SURFACE-NORMAL and SAMPLE-DIRECTION (vector from SURFACE-POSITION to SAMPLE-POSITION)
				float NdotS = max(dot(Normal, sampleDir), 0.0);
				// distance between SURFACE-POSITION and SAMPLE-POSITION
				float VPdistSP = distance(FragPos, samplePos);

				// a = distance function
				float a = 1.0 - smoothstep(ssaoDistanceThreshold, ssaoDistanceThreshold * 2, VPdistSP);
				// b = dot-Product
				float b = NdotS;

				ambientOcclusion += (a * b);
			}

		    ambientOcclusion /= sample_count;
		}

        mat3 viewM = mat3(ubo.ViewMatrix*ubo.ModelMatrix);
		vec3 viewDir = normalize(viewM*(ubo.CamPosition - FragPos));

		for(int i = 0; i < numLights; ++i)
		{
            vec3 lightDir = normalize(viewM*(lights[i].Position.rgb - FragPos));
            vec3 halfway = normalize(lightDir + viewDir);
            float distance = length(lights[i].Position.rgb - FragPos);
            float lightAttenuation = 1.0 / (1.0 + lights[i].Linear * distance + lights[i].Quadratic * distance * distance);

		    if(reflectanceModel == 0) {
		        // Diffuse
             	vec3 diffuse = max(dot(Normal, lightDir), 0.0) * lights[i].Intensity * Albedo.rgb * lights[i].Color.rgb * (1.0f - ambientOcclusion);
             	float spec = pow(max(dot(Normal, halfway), 0.0), Specular);
             	vec3 specular = lights[i].Color.rgb * spec;

             	diffuse *= lightAttenuation;
             	specular *= lightAttenuation;
             	lighting += diffuse + specular;
		    }
		    // Oren-Nayar model
		    if(reflectanceModel == 1) {

            	float NdotL = dot(Normal, lightDir);
            	float NdotV = dot(Normal, viewDir);

            	float angleVN = acos(NdotV);
            	float angleLN = acos(NdotL);

            	float alpha = max(angleVN, angleLN);
            	float beta = min(angleVN, angleLN);
            	float gamma = dot(viewDir - Normal*dot(viewDir, Normal), lightDir - Normal*dot(lightDir, Normal));

            	float roughness = 0.75;

            	float roughnessSquared = roughness*roughness;

            	float A = 1.0 - 0.5 * ( roughnessSquared / (roughnessSquared + 0.57));
            	float B = 0.45 * (roughnessSquared / (roughnessSquared + 0.09));
            	float C = sin(alpha)*tan(beta);

            	float L1 = max(0.0, NdotL) * (A + B * max(0.0, gamma) * C);

                vec3 inputColor = lightAttenuation*lights[i].Intensity * lights[i].Color.rgb * Albedo.rgb * (1.0f - ambientOcclusion);

            	vec3 diffuse = inputColor * L1;
            	vec3 specular = vec3(0.0);

            	specular *= lightAttenuation*specular;
            	lighting += diffuse + specular;
            }
            // Cook-Torrance
            if(reflectanceModel == 2) {
                float metallic = 1.0;
                vec3 F0 = vec3(0.04);
                F0 = mix(F0, Albedo.rgb, metallic);

                float roughness = 1.0 - Specular;

                float NDF = GGXDistribution(Normal, halfway, roughness);
                float G = GeometrySmith(Normal, viewDir, lightDir, roughness);
                vec3 F = FresnelSchlick(max(dot(halfway, viewDir), 0.0), F0);

                vec3 BRDF = (NDF * G * F)/(4 * max(dot(viewDir, Normal), 0.0) * max(dot(lightDir, Normal), 0.0) + 0.001);

                vec3 kS = F;
                vec3 kD = (vec3(1.0) - kS);
                kD *= 1.0 - metallic;

                float NdotL = max(dot(Normal, lightDir), 0.0);
                vec3 radiance = lights[i].Intensity * lights[i].Color.rgb * lightAttenuation;

                lighting += (kD * Albedo.rgb / PI + BRDF) * radiance * NdotL;
            }
		}

		FragColor = vec4(lighting, 1.0);
	} else {
		vec2 newTexCoord;
		// color
		if(textureCoord.x < 0.25 && textureCoord.y < 0.5 ) {
			FragColor = Albedo;
		}
		// specular
		if(textureCoord.x > 0.25 && textureCoord.x < 0.5 && textureCoord.y < 0.5) {
		    FragColor = vec4(Specular, Specular, Specular, 1.0);
		}
		// depth
		if(textureCoord.x > 0.5 && textureCoord.y < 0.5) {
		    float near = 0.5f;
		    float far = 1000.0f;
		    vec3 linearizedDepth = vec3((2.0f * near) / (far + near - Depth * (far - near)));
			FragColor = vec4(linearizedDepth, 1.0f);
		}
		// normal
		if(textureCoord.x > 0.5 && textureCoord.y > 0.5) {
			FragColor = vec4(Normal, 1.0f);
		}
		// position
		if(textureCoord.x < 0.5 && textureCoord.y > 0.5) {
			FragColor = vec4(FragPos, 1.0f);
		}
	}
}
