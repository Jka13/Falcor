# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "C:/Program Files/CMake/bin/cmake.exe"

# The command to remove a file.
RM = "C:/Program Files/CMake/bin/cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:/Users/jonas/Documents/Falcor/Falcor

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:/Users/jonas/Documents/Falcor/Falcor/cmake

# Include any dependencies generated for this target.
include Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/progress.make

# Include the compile flags for this target's objects.
include Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/flags.make

bin/shaders/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.rt.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.rt.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "MinimalPathTracerShadowMap: Copying shader MinimalPathTracerShadowMap.rt.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.rt.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.rt.slang

Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/flags.make
Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/includes_CXX.rsp
Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.cpp
Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj -MF CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj.d -o CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.cpp

Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.cpp > CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.i

Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.cpp -o CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.s

# Object files for target MinimalPathTracerShadowMap
MinimalPathTracerShadowMap_OBJECTS = \
"CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj"

# External object files for target MinimalPathTracerShadowMap
MinimalPathTracerShadowMap_EXTERNAL_OBJECTS =

bin/plugins/MinimalPathTracerShadowMap.dll: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/MinimalPathTracerShadowMap.cpp.obj
bin/plugins/MinimalPathTracerShadowMap.dll: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/build.make
bin/plugins/MinimalPathTracerShadowMap.dll: Source/Falcor/libFalcor.dll.a
bin/plugins/MinimalPathTracerShadowMap.dll: external/fmt/libfmt.a
bin/plugins/MinimalPathTracerShadowMap.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/plugins/MinimalPathTracerShadowMap.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/plugins/MinimalPathTracerShadowMap.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/plugins/MinimalPathTracerShadowMap.dll: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/linkLibs.rsp
bin/plugins/MinimalPathTracerShadowMap.dll: Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library ../../../bin/plugins/MinimalPathTracerShadowMap.dll"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/MinimalPathTracerShadowMap.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/MinimalPathTracerShadowMap.dir/objects.a @CMakeFiles/MinimalPathTracerShadowMap.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && C:/Strawberry/c/bin/c++.exe -shared -o ../../../bin/plugins/MinimalPathTracerShadowMap.dll -Wl,--out-implib,libMinimalPathTracerShadowMap.dll.a -Wl,--major-image-version,0,--minor-image-version,0 -Wl,--whole-archive CMakeFiles/MinimalPathTracerShadowMap.dir/objects.a -Wl,--no-whole-archive @CMakeFiles/MinimalPathTracerShadowMap.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/build: bin/plugins/MinimalPathTracerShadowMap.dll
.PHONY : Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/build

Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap && $(CMAKE_COMMAND) -P CMakeFiles/MinimalPathTracerShadowMap.dir/cmake_clean.cmake
.PHONY : Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/clean

Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/depend: bin/shaders/RenderPasses/MinimalPathTracerShadowMap/MinimalPathTracerShadowMap.rt.slang
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/MinimalPathTracerShadowMap C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/RenderPasses/MinimalPathTracerShadowMap/CMakeFiles/MinimalPathTracerShadowMap.dir/depend

