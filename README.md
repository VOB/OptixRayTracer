# OptixRayTracer

Requirements to run code:

- Visual Studio (preferred 2013. Earlier works, but 2015 is not supported by NVidia OptiX 3.9)

- CUDA Toolkit

- CMake 3.0 minimum


Build instructions

1. Before building, you MUST import cloth shader by performing the following commands:
	git submodule init
	git submodule update

2. Start up cmake-gui from the Start Menu.

3. Select the "downloadeddirectory/SDK" for the source code location.

4. Create a build directory that isn't the same as the source directory, 
and set this as your build location.

5. Press configure and click OK. Then press configure again when it is done.

6. Press Generate.

Congratulations! You should now have a project set up in your build directory. 
Open the Optix_Project file with VS and enjoy.
