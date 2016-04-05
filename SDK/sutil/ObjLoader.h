
/*
 * Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once
#include <sutil.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <glm.h>
#include <string>

//-----------------------------------------------------------------------------
// 
//  ObjLoader class declaration 
//
//-----------------------------------------------------------------------------
class ObjLoader
{
public:
  SUTILAPI ObjLoader( const char* filename,                 // Model filename
                      optix::Context context,               // Context for RT object creation
                      optix::GeometryGroup geometrygroup ); // Empty geom group to hold model
  SUTILAPI ObjLoader( const char* filename,
                      optix::Context context,
                      optix::GeometryGroup geometrygroup,
                      optix::Material material,             // Material override
                      bool force_load_material_params = false); // Set obj_material params even though material is overridden

  SUTILAPI ~ObjLoader() {} // makes sure CRT objects are destroyed on the correct heap

  SUTILAPI void setIntersectProgram( optix::Program program );
  SUTILAPI void load();
  SUTILAPI void load( const optix::Matrix4x4& transform );

  SUTILAPI optix::Aabb getSceneBBox()const { return _aabb; }
  SUTILAPI optix::Buffer getLightBuffer()const { return _light_buffer; }

  SUTILAPI static bool isMyFile( const char* filename );

private:

  struct MatParams
  {
    optix::float3 emissive;
    optix::float3 reflectivity;
    float  phong_exp;
    int    illum;
    optix::TextureSampler ambient_map;
    optix::TextureSampler diffuse_map;
    optix::TextureSampler specular_map;
  };

  void createMaterial();
  void createGeometryInstances( GLMmodel* model,
                                optix::Program mesh_intersect,
                                optix::Program mesh_bbox );
  void loadVertexData( GLMmodel* model, const optix::Matrix4x4& transform );
  void createMaterialParams( GLMmodel* model );
  void loadMaterialParams( optix::GeometryInstance gi, unsigned int index );
  void createLightBuffer( GLMmodel* model );

  std::string            _pathname;
  std::string            _filename;
  optix::Context         _context;
  optix::GeometryGroup   _geometrygroup;
  optix::Buffer          _vbuffer;
  optix::Buffer          _nbuffer;
  optix::Buffer          _tbuffer;
  optix::Material        _material;
  optix::Program         _intersect_program;
  optix::Buffer          _light_buffer;
  bool                   _have_default_material;
  bool                   _force_load_material_params;
  optix::Aabb            _aabb;
  std::vector<MatParams> _material_params;
};


