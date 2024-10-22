
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

#include <ImageLoader.h>
#include <PPMLoader.h>
#include <HDRLoader.h>
#include <fstream>


//-----------------------------------------------------------------------------
//  
//  Utility functions 
//
//-----------------------------------------------------------------------------

SUTILAPI optix::TextureSampler loadTexture( optix::Context context,
                                            const std::string& filename,
                                            const optix::float3& default_color )
{
  bool IsHDR = false;
  size_t len = filename.length();
  if(len >= 3) {
    IsHDR = (filename[len-3] == 'H' || filename[len-3] == 'h') &&
      (filename[len-2] == 'D' || filename[len-2] == 'd') &&
      (filename[len-1] == 'R' || filename[len-1] == 'r');
  }
  if(IsHDR)
    return loadHDRTexture(context, filename, default_color);
  else
    return loadPPMTexture(context, filename, default_color);
}

SUTILAPI optix::Buffer loadCubeBuffer( optix::Context context,
                                       const std::vector<std::string>& filenames )
{
  return loadPPMCubeBuffer(context, filenames);
}
