
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

	unsigned int m_width;
	unsigned int m_height;
	std::string   texture_path;
	std::string  m_ptx_path;
};


void OptixProject::initScene(InitialCameraData& camera_data)
{
	// set up path to ptx file associated with tutorial number
	std::stringstream ss;
	ss << "cudaFile.cu";
	m_ptx_path = ptxpath("optixProject", ss.str());

	// context 
	m_context->setRayTypeCount(2);
	m_context->setEntryPointCount(1);
	m_context->setStackSize(4640);

	m_context["max_depth"]->setInt(100);
	m_context["radiance_ray_type"]->setUint(0);
	m_context["shadow_ray_type"]->setUint(1);
	m_context["frame_number"]->setUint(0u);
	m_context["scene_epsilon"]->setFloat(1.e-3f);
	m_context["importance_cutoff"]->setFloat(0.01f);
	m_context["ambient_light_color"]->setFloat(0.3f, 0.33f, 0.28f);

	m_context["output_buffer"]->set(createOutputBuffer(RT_FORMAT_UNSIGNED_BYTE4, m_width, m_height));

	// Ray gen program
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
	camera_data = InitialCameraData(make_float3(7.0f, 9.2f, -6.0f), // eye
		make_float3(0.0f, 0.0f, 0.0f), // lookat
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

	// Prepare to run
	m_context->validate();
	m_context->compile();
}


Buffer OptixProject::getOutputBuffer()
{
	return m_context["output_buffer"]->getBuffer();
}


void OptixProject::trace(const RayGenCameraData& camera_data)
{
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

void OptixProject::createGeometry() //-----------------------------------------------------------------// DO NOT CREATE ANY GEOMETRY, we want to import a scene
{
	std::string box_ptx(ptxpath("optixProject", "box.cu"));
	Program box_bounds = m_context->createProgramFromPTXFile(box_ptx, "box_bounds");
	Program box_intersect = m_context->createProgramFromPTXFile(box_ptx, "box_intersect");
	
	// Create box
	Geometry box = m_context->createGeometry();
	box->setPrimitiveCount(1u);
	box->setBoundingBoxProgram(box_bounds);
	box->setIntersectionProgram(box_intersect);
	box["boxmin"]->setFloat(-2.0f, 0.0f, -2.0f);
	box["boxmax"]->setFloat(2.0f, 7.0f, 2.0f);
	
	// Floor geometry
	std::string pgram_ptx(ptxpath("optixProject", "parallelogram.cu"));
	Geometry parallelogram = m_context->createGeometry();
	parallelogram->setPrimitiveCount(1u);
	parallelogram->setBoundingBoxProgram(m_context->createProgramFromPTXFile(pgram_ptx, "bounds"));
	parallelogram->setIntersectionProgram(m_context->createProgramFromPTXFile(pgram_ptx, "intersect"));
	float3 anchor = make_float3(-64.0f, 0.01f, -64.0f);
	float3 v1 = make_float3(128.0f, 0.0f, 0.0f);
	float3 v2 = make_float3(0.0f, 0.0f, 128.0f);
	float3 normal = cross(v2, v1);
	normal = normalize(normal);
	float d = dot(normal, anchor);
	v1 *= 1.0f / dot(v1, v1);
	v2 *= 1.0f / dot(v2, v2);
	float4 plane = make_float4(normal, d);
	parallelogram["plane"]->setFloat(plane);
	parallelogram["v1"]->setFloat(v1);
	parallelogram["v2"]->setFloat(v2);
	parallelogram["anchor"]->setFloat(anchor);
	
	// Materials
	std::string box_chname = "cloth_closest_hit_radiance";

	Material box_matl = m_context->createMaterial();
	Program box_ch = m_context->createProgramFromPTXFile(m_ptx_path, box_chname);
	box_matl->setClosestHitProgram(0, box_ch);
	
	Program box_ah = m_context->createProgramFromPTXFile(m_ptx_path, "any_hit_shadow");
	box_matl->setAnyHitProgram(1, box_ah);

    wcWeaveParameters weave_params;
    // --- Woven Cloth parameters --
    char *weave_pattern_filename = "34697.wif";
	weave_params.uscale = 4.f;
	weave_params.vscale = 4.f;
	weave_params.umax   = 0.5f;
	weave_params.psi    = 0.5f;
    weave_params.alpha = 0.01f;
    weave_params.beta = 4.f;
    weave_params.delta_x = 0.6f;
    weave_params.intensity_fineness = 0.f;
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
    box_matl["wc_specular_strength"]->setFloat(specular_strength);
    box_matl["wc_parameters"]->setUserData(sizeof(wcWeaveParameters),&weave_params);
    box_matl["wc_pattern"]->setUserData(sizeof(PatternEntry)*pattern_size,weave_params.pattern_entry);
	

	Material chair_matl = m_context->createMaterial();
	Program chair_ch = m_context->createProgramFromPTXFile(m_ptx_path, "floor_closest_hit_radiance");
	chair_matl->setClosestHitProgram(0, chair_ch);
	
	Program ah = m_context->createProgramFromPTXFile(m_ptx_path, "any_hit_shadow");
	chair_matl->setAnyHitProgram(1, ah);
	
	chair_matl["Ka"]->setFloat(0.f, 0.f, 0.f);
	chair_matl["Kd"]->setFloat(.2f, .2f, .2f);
	chair_matl["Ks"]->setFloat(0.2f, 0.2f, 0.2f);
	chair_matl["reflectivity"]->setFloat(0.1f, 0.1f, 0.1f);
	chair_matl["reflectivity_n"]->setFloat(0.01f, 0.01f, 0.01f);
	chair_matl["phong_exp"]->setFloat(88);
	chair_matl["tile_v0"]->setFloat(0.25f, 0, .15f);
	chair_matl["tile_v1"]->setFloat(-.15f, 0, 0.25f);
	chair_matl["crack_color"]->setFloat(0.1f, 0.1f, 0.1f);
	chair_matl["crack_width"]->setFloat(0.f);

    Material floor_matl = m_context->createMaterial();
	Program floor_ch = m_context->createProgramFromPTXFile(m_ptx_path, "only_shadows_closest_hit_radiance");
	floor_matl->setClosestHitProgram(0, floor_ch);

	floor_matl->setAnyHitProgram(1, ah);

    // ---- Load obj files ----

    const char *obj_names[] = {
        "cloth_on_chair.obj",
        "chair.obj",
        "floor.obj",
    };

    optix::Material obj_materials[] = {
        box_matl,
        chair_matl,
        floor_matl
    };

    Group root_group = m_context->createGroup();
    for(int i=0;i<sizeof(obj_names)/sizeof(char*);i++){
	    GeometryGroup geometrygroup = m_context->createGeometryGroup();
	    std::string path = texpath(obj_names[i]);
	    ObjLoader loader(path.c_str(), m_context, geometrygroup, obj_materials[i]);
	    loader.load();
        root_group->addChild<GeometryGroup>(geometrygroup);
    }

    root_group->setAcceleration(m_context->createAcceleration("NoAccel", "NoAccel"));
	
	m_context["top_object"]->set(root_group);
	m_context["top_shadower"]->set(root_group);
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
		GLUTDisplay::run(title.str(), &scene);
	}
	catch (Exception& e){
		sutilReportError(e.getErrorString().c_str());
		exit(1);
	}
	return 0;
}
