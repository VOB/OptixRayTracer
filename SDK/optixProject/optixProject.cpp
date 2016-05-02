
// 0 - normal shader
// 1 - lambertian
// 2 - specular
// 3 - shadows
// 4 - reflections
// 5 - miss
// 6 - schlick
// 7 - procedural texture on floor
// 8 - LGRustyMetal
// 9 - intersection
// 10 - anyhit
// 11 - camera


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_namespace.h>
#include <iostream>
#include <GLUTDisplay.h>
#include <ImageLoader.h>
#include "commonStructs.h"
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <math.h>
#include <sutil.h>
#include "PPMLoader.h"
#include <OptiXMesh.h>
#include <ObjLoader.h>

#include "woven_cloth.h"
#include "random.h"

using namespace optix;

static float rand_range(float min, float max)
{
	return min + (max - min) * (float)rand() / (float)RAND_MAX;
}


//-----------------------------------------------------------------------------
// 
// Whitted Scene
//
//-----------------------------------------------------------------------------

class OptixProject : public SampleScene
{
public:
	OptixProject(const std::string& texture_path)
		: SampleScene(), m_width(1080u), m_height(720u), texture_path(texture_path)
	{}

	// From SampleScene
	void   initScene(InitialCameraData& camera_data);
	void   trace(const RayGenCameraData& camera_data);
	void   doResize(unsigned int width, unsigned int height);
	void   setDimensions(const unsigned int w, const unsigned int h) { m_width = w; m_height = h; }
	Buffer getOutputBuffer();

private:
	std::string texpath(const std::string& base);
	void createGeometry();
	void updateGeometry();
	void initAnimation();

	unsigned int m_width;
	unsigned int m_height;
	std::string   texture_path;
	std::string  m_ptx_path;
	GeometryGroup geometrygroup;
	float3*       m_vertices;
	Material glass_matl;

	Buffer m_rnd_seeds;
};


void OptixProject::initScene(InitialCameraData& camera_data)
{

	std::stringstream ss;
	ss << "cudaFile.cu";
	m_ptx_path = ptxpath("optixProject", ss.str());

	// Setup state
	m_context->setRayTypeCount(2);
	m_context->setEntryPointCount(1);
	m_context->setStackSize(4640);

	//Context variables
	m_context["max_depth"]->setInt(16);
	m_context["radiance_ray_type"]->setUint(0);
	m_context["shadow_ray_type"]->setUint(1);
	m_context["scene_epsilon"]->setFloat(1.e-3f);
	m_context["importance_cutoff"]->setFloat(0.01f);
	m_context["ambient_light_color"]->setFloat(0.3f, 0.33f, 0.28f);

	//Shadow ray modifiers
	m_context["shadow_samples"]->setUint(2);
	m_context["light_radius"]->setUint(2);

	//Rendering variables 
	m_context["frame_number"]->setUint(0u);
	m_context["output_buffer"]->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height));

	//Anti aliasing variables
	m_context["jitter_factor"]->setFloat(1.0f);
	m_context["frame"]->setUint(0u);

	m_rnd_seeds = m_context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL, RT_FORMAT_UNSIGNED_INT, m_width, m_height);
	m_context["rnd_seeds"]->setBuffer(m_rnd_seeds);
	unsigned int* seeds = static_cast<unsigned int*>(m_rnd_seeds->map());
	fillRandBuffer(seeds, m_width*m_height);
	m_rnd_seeds->unmap();

	// Ray gen program
	m_context["super_samples"]->setUint(1);
	std::string camera_name = "pinhole_camera";

	Program ray_gen_program = m_context->createProgramFromPTXFile(m_ptx_path, camera_name);
	m_context->setRayGenerationProgram(0, ray_gen_program);

	// Exception / miss programs
	Program exception_program = m_context->createProgramFromPTXFile(m_ptx_path, "exception");
	m_context->setExceptionProgram(0, exception_program);
	m_context["bad_color"]->setFloat(0.0f, 1.0f, 0.0f);

	std::string miss_name = "envmap_miss";
	m_context->setMissProgram(0, m_context->createProgramFromPTXFile(m_ptx_path, miss_name));
	const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
	m_context["envmap"]->setTextureSampler(loadTexture(m_context, texpath("Photo-studio-with-umbrella.hdr"), default_color));
	m_context["bg_color"]->setFloat(make_float3(1.f, 1.f, 1.f));

	// Lights
	BasicLight lights[] = {
		{ make_float3(-5.0f, 50.0f, -16.0f), make_float3(0.4f, 0.4f, 0.5f), 1 },
        { make_float3(25.0f, 50.0f, -16.0f), make_float3(0.5f, 0.5f, 0.4f), 1 }
	};

	Buffer light_buffer = m_context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();

	m_context["lights"]->set(light_buffer);

	// Set up camera
	camera_data = InitialCameraData(make_float3(9.0f, 10.0f, -2.0f), // eye
		make_float3(0.0f, 4.0f, 0.0f), // lookat
		make_float3(0.0f, 1.0f, 0.0f), // up
		60.0f);                          // vfov

	m_context["eye"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["U"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["V"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));
	m_context["W"]->setFloat(make_float3(0.0f, 0.0f, 0.0f));

	// 3D solid noise buffer, 1 float channel, all entries in the range [0.0, 1.0].
	srand(0); // Make sure the pseudo random numbers are the same every run.

	int tex_width = 64;

	int tex_height = 64;
	int tex_depth = 64;
	Buffer noiseBuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT, tex_width, tex_height, tex_depth);
	float *tex_data = (float *)noiseBuffer->map();

	// Random noise in range [0, 1]
	for (int i = tex_width * tex_height * tex_depth; i > 0; i--) {
		// One channel 3D noise in [0.0, 1.0] range.
		*tex_data++ = rand_range(0.0f, 1.0f);
	}
	noiseBuffer->unmap();


	// Noise texture sampler
	TextureSampler noiseSampler = m_context->createTextureSampler();

	noiseSampler->setWrapMode(0, RT_WRAP_REPEAT);
	noiseSampler->setWrapMode(1, RT_WRAP_REPEAT);
	noiseSampler->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
	noiseSampler->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
	noiseSampler->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
	noiseSampler->setMaxAnisotropy(1.0f);
	noiseSampler->setMipLevelCount(1);
	noiseSampler->setArraySize(1);
	noiseSampler->setBuffer(0, 0, noiseBuffer);

	m_context["noise_texture"]->setTextureSampler(noiseSampler);

	// Populate scene hierarchy
	createGeometry();
	initAnimation();

	// Prepare to run
	m_context->validate();
	m_context->compile();
}

void OptixProject::initAnimation()
{

	GeometryInstance geometryInstance = geometrygroup->getChild(0);
	Geometry geometry = geometryInstance->getGeometry();

	/*
	All that we want to do here is to copy
	the original vertex positions we get from
	the OptiXMesh to an array. We use this
	array to always have access to the
	original vertex position; the values in the
	OptiXMesh buffer will be altered per frame.
	*/

	//Query vertex buffer
	Buffer vertexBuffer = geometry["vertex_buffer"]->getBuffer();

	//Query number of vertices in the buffer
	RTsize numVertices;
	vertexBuffer->getSize(numVertices);

	//Get a pointer to the buffer data
	float3* original_vertices = (float3*)vertexBuffer->map();

	//Allocate our storage array and copy values
	m_vertices = new float3[numVertices];
	memcpy(m_vertices, original_vertices, numVertices * sizeof(float3));

	//Unmap buffer
	vertexBuffer->unmap();
}


Buffer OptixProject::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}


void OptixProject::trace(const RayGenCameraData& camera_data)
{
	

	//updateGeometry();
	static float t = 0;
	m_context["eye"]->setFloat(camera_data.eye);
	m_context["U"]->setFloat(camera_data.U);
	m_context["V"]->setFloat(camera_data.V);
	m_context["W"]->setFloat(camera_data.W);

	Buffer buffer = m_context["output_buffer"]->getBuffer();
	RTsize buffer_width, buffer_height;
	buffer->getSize(buffer_width, buffer_height);

	m_context->launch(0, static_cast<unsigned int>(buffer_width),
		static_cast<unsigned int>(buffer_height));
}

void OptixProject::updateGeometry() {
	GeometryInstance test = geometrygroup->getChild(0);
	Geometry test2 = test->getGeometry();


	/*
	All we want to do here is to add a simple sin(x) offset
	to the vertices y-position.
	*/

	Buffer vertexBuffer = test2["vertex_buffer"]->getBuffer();
	float3* new_vertices = (float3*)vertexBuffer->map();

	RTsize numVertices;
	vertexBuffer->getSize(numVertices);

	static float t = 0.0f;

	//We don't have to set x and z here in this example
	for (unsigned int v = 0; v < numVertices; v++)
	{
		new_vertices[v].y = m_vertices[v].y + (sinf(m_vertices[v].x / 0.3f * 3.0f + t) * 0.3f * 0.7f);
	}

	t += 0.1f;

	vertexBuffer->unmap();

	/*
	Vertices are changed now; we have to tell the
	corresponding acceleration structure that it
	has to be rebuilt.

	Mark the accel structure and geometry as dirty.
	*/
	test2->markDirty();
	geometrygroup->getAcceleration()->markDirty();
	
}

void OptixProject::doResize(unsigned int width, unsigned int height)
{
	// output buffer handled in SampleScene::resize
}

std::string OptixProject::texpath(const std::string& base)
{
	return texture_path + "/" + base;
}

float4 make_plane(float3 n, float3 p)
{
	n = normalize(n);
	float d = -dot(n, p);
	return make_float4(n, d);
}

struct ObjFile
{
    const char *filename;
    optix::Material material;
    float transform[4*4];
};

void OptixProject::createGeometry() //-----------------------------------------------------------------// DO NOT CREATE ANY GEOMETRY, we want to import a scene
{
	
	// Materials
	std::string cloth_chname = "cloth_closest_hit_radiance";

	Material cloth_matl = m_context->createMaterial();
	Program cloth_ch = m_context->createProgramFromPTXFile(m_ptx_path, cloth_chname);
	cloth_matl->setClosestHitProgram(0, cloth_ch);
	
	Program cloth_ah = m_context->createProgramFromPTXFile(m_ptx_path, "any_hit_shadow");
	cloth_matl->setAnyHitProgram(1, cloth_ah);

    wcWeaveParameters weave_params;
    // --- Woven Cloth parameters --
    char *weave_pattern_filename = "nordvalla.weave";
	weave_params.uscale = 1000.f;
	weave_params.vscale = 1000.f;
	weave_params.umax   = 0.5f;
	weave_params.psi    = 0.5f;
    weave_params.alpha = 0.01f;
    weave_params.beta = 4.f;
    weave_params.delta_x = 0.3f;
    weave_params.intensity_fineness = 16.f;
    weave_params.yarnvar_amplitude = 1.f;
    weave_params.yarnvar_xscale = 1.f;
    weave_params.yarnvar_yscale = 1.f;
    weave_params.yarnvar_persistance = 1.f;
    weave_params.yarnvar_octaves = 8.f;
    float specular_strength = 0.3;
    // -----------------------------

    std::string weave_pattern_path = texpath(weave_pattern_filename);
    wcWeavePatternFromFile(&weave_params,weave_pattern_path.c_str());
    int pattern_size = weave_params.pattern_width*weave_params.pattern_height;
    cloth_matl["wc_specular_strength"]->setFloat(specular_strength);
    cloth_matl["wc_parameters"]->setUserData(sizeof(wcWeaveParameters),&weave_params);
    cloth_matl["wc_pattern"]->setUserData(sizeof(PatternEntry)*pattern_size,weave_params.pattern_entry);
	

	Material chair_matl = m_context->createMaterial();
	Program chair_ch = m_context->createProgramFromPTXFile(m_ptx_path, "chair_closest_hit_radiance");
	chair_matl->setClosestHitProgram(0, chair_ch);
	
	Program ah = m_context->createProgramFromPTXFile(m_ptx_path, "any_hit_shadow");
	chair_matl->setAnyHitProgram(1, ah);
	
	chair_matl["Ka"]->setFloat(0.7f, 0.5f, 0.3f);
	chair_matl["Kd"]->setFloat(.2f, .2f, .2f);
	chair_matl["Ks"]->setFloat(0.2f, 0.2f, 0.2f);
	chair_matl["reflectivity"]->setFloat(0.1f, 0.1f, 0.1f);
	chair_matl["reflectivity_n"]->setFloat(0.01f, 0.01f, 0.01f);
	chair_matl["phong_exp"]->setFloat(88);

    Material floor_matl = m_context->createMaterial();
	Program floor_ch = m_context->createProgramFromPTXFile(m_ptx_path, "only_shadows_closest_hit_radiance");
	floor_matl->setClosestHitProgram(0, floor_ch);

	floor_matl->setAnyHitProgram(1, ah);

	Material metal_matl = m_context->createMaterial();
	Program metal_ch = m_context->createProgramFromPTXFile(m_ptx_path, "metal_closest_hit_radiance");
	metal_matl->setClosestHitProgram(0, metal_ch);
	
	Program metal_ah = m_context->createProgramFromPTXFile(m_ptx_path, "any_hit_shadow");
	metal_matl->setAnyHitProgram(1, metal_ah);
	
	metal_matl["Ka"]->setFloat(0.3f, 0.3f, 0.3f);
	metal_matl["Kd"]->setFloat(0.6f, 0.7f, 0.8f);
	metal_matl["Ks"]->setFloat(0.8f, 0.9f, 0.8f);
	metal_matl["phong_exp"]->setFloat(88);
	metal_matl["reflectivity_n"]->setFloat(0.2f, 0.2f, 0.2f);

	// Glass material
	
	//if (chull.get()) {
		Program glass_ch = m_context->createProgramFromPTXFile(m_ptx_path, "glass_closest_hit_radiance");

		std::string glass_ahname = "glass_any_hit_shadow";

		Program glass_ah = m_context->createProgramFromPTXFile(m_ptx_path, glass_ahname);
		glass_matl = m_context->createMaterial();
		glass_matl->setClosestHitProgram(0, glass_ch);
		glass_matl->setAnyHitProgram(1, glass_ah);

		glass_matl["importance_cutoff"]->setFloat(1e-2f);
		glass_matl["cutoff_color"]->setFloat(0.34f, 0.55f, 0.85f);
		glass_matl["fresnel_exponent"]->setFloat(3.0f);
		glass_matl["fresnel_minimum"]->setFloat(0.1f);
		glass_matl["fresnel_maximum"]->setFloat(1.0f);
		glass_matl["refraction_index"]->setFloat(1.4f);
		glass_matl["refraction_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["reflection_color"]->setFloat(1.0f, 1.0f, 1.0f);
		glass_matl["refraction_maxdepth"]->setInt(100);
		glass_matl["reflection_maxdepth"]->setInt(100);
		float3 extinction = make_float3(.80f, .89f, .75f);
		glass_matl["extinction_constant"]->setFloat(log(extinction.x), log(extinction.y), log(extinction.z));
		glass_matl["shadow_attenuation"]->setFloat(0.4f, 0.7f, 0.4f);
	//}

	geometrygroup = m_context->createGeometryGroup();

    struct ObjFile objs[] = {
		
        {"floor.obj", floor_matl, 
           {1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 0.0f,
		    0.0f, 0.0f, 0.0f, 1.0f}
        },

		{ "bunny_uv.obj", glass_matl,
			{ 10.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 10.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 10.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 10.0f }
		},

		{"cognacglass.obj", metal_matl, 
            {0.16f, 0.0f, 0.0f, 0.0f,
		    0.0f, 0.16f, 0.0f, 4.55f,
		    0.0f, 0.0f, 0.16f, 0.0f,
		    0.0f, 0.0f, 0.0f, 0.16f}
        },
        
        {"chair.obj", chair_matl, 
           {1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 0.0f,
		    0.0f, 0.0f, 0.0f, 1.0f}
        },
        {"cloth_on_chair.obj", cloth_matl, 
           {1.0f, 0.0f, 0.0f, 0.0f,
		    0.0f, 1.0f, 0.0f, 0.0f,
		    0.0f, 0.0f, 1.0f, 0.0f,
		    0.0f, 0.0f, 0.0f, 1.0f}
        }
    };

    for(int i=0;i<sizeof(objs)/sizeof(ObjFile);i++){
        OptiXMesh mesh(m_context, geometrygroup, objs[i].material, m_accel_desc);
	    Matrix4x4 matx0 = Matrix4x4(objs[i].transform);
	    mesh.setLoadingTransform(matx0);
	    mesh.loadBegin_Geometry(texpath(objs[i].filename));
	    mesh.loadFinish_Materials();
    }

	m_context["top_object"]->set(geometrygroup);
	m_context["top_shadower"]->set(geometrygroup);
}


//-----------------------------------------------------------------------------
//
// Main driver
//
//-----------------------------------------------------------------------------

void printUsageAndExit(const std::string& argv0, bool doExit = true)
{
	std::cerr
		<< "Usage  : " << argv0 << " [options]\n"
		<< "App options:\n"
		<< "  -h  | --help                               Print this usage message\n"
		<< "  -t  | --texture-path <path>                Specify path to texture directory\n"
		<< "        --dim=<width>x<height>               Set image dimensions\n"
		<< std::endl;
	GLUTDisplay::printUsage();

	if (doExit) exit(1);
}


int main(int argc, char** argv)
{
	GLUTDisplay::init(argc, argv);

	unsigned int width = 1080u, height = 720u;

	std::string texture_path;
	
	if( !GLUTDisplay::isBenchmark() ) printUsageAndExit( argv[0], false );
	
	if (texture_path.empty()) {
		texture_path = std::string(sutilSamplesDir()) + "/optixProject/data";
	}
	//--------------------------------------------//
	//--------------------------------------------//

	std::stringstream title;
	title << "Nice stuff yo";
	try {
		
		OptixProject scene(texture_path);
		scene.setDimensions(width, height);
		GLUTDisplay::run(title.str(), &scene, GLUTDisplay::CDAnimated);
	}
	catch (Exception& e){
		sutilReportError(e.getErrorString().c_str());
		exit(1);
	}
	return 0;
}
