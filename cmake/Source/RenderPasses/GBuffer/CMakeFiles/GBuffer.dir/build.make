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
include Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/progress.make

# Include the compile flags for this target's objects.
include Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make

bin/shaders/RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "GBuffer: Copying shader GBuffer/DepthPass.3d.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang

bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferHelpers.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferHelpers.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "GBuffer: Copying shader GBuffer/GBufferHelpers.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferHelpers.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferHelpers.slang

bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "GBuffer: Copying shader GBuffer/GBufferRaster.3d.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang

bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "GBuffer: Copying shader GBuffer/GBufferRT.cs.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang

bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "GBuffer: Copying shader GBuffer/GBufferRT.rt.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang

bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "GBuffer: Copying shader GBuffer/GBufferRT.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.slang

bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "GBuffer: Copying shader VBuffer/VBufferRaster.3d.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang

bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "GBuffer: Copying shader VBuffer/VBufferRT.cs.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang

bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "GBuffer: Copying shader VBuffer/VBufferRT.rt.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang

bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "GBuffer: Copying shader VBuffer/VBufferRT.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.slang

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBufferBase.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj -MF CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj.d -o CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBufferBase.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/GBufferBase.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBufferBase.cpp > CMakeFiles/GBuffer.dir/GBufferBase.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/GBufferBase.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBufferBase.cpp -o CMakeFiles/GBuffer.dir/GBufferBase.cpp.s

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBuffer.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj -MF CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj.d -o CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBuffer.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBuffer.cpp > CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBuffer.cpp -o CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.s

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj -MF CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj.d -o CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.cpp > CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.cpp -o CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.s

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj -MF CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj.d -o CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cpp > CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/GBuffer/GBufferRT.cpp -o CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.s

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj -MF CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj.d -o CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.cpp > CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRaster.cpp -o CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.s

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/flags.make
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/includes_CXX.rsp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cpp
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj -MF CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj.d -o CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cpp

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cpp > CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.i

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer/VBuffer/VBufferRT.cpp -o CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.s

# Object files for target GBuffer
GBuffer_OBJECTS = \
"CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj" \
"CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj" \
"CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj" \
"CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj" \
"CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj" \
"CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj"

# External object files for target GBuffer
GBuffer_EXTERNAL_OBJECTS =

bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBufferBase.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBuffer.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRaster.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/GBuffer/GBufferRT.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRaster.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/VBuffer/VBufferRT.cpp.obj
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/build.make
bin/plugins/GBuffer.dll: Source/Falcor/libFalcor.dll.a
bin/plugins/GBuffer.dll: external/fmt/libfmt.a
bin/plugins/GBuffer.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/plugins/GBuffer.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/plugins/GBuffer.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/linkLibs.rsp
bin/plugins/GBuffer.dll: Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Linking CXX shared library ../../../bin/plugins/GBuffer.dll"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/GBuffer.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/GBuffer.dir/objects.a @CMakeFiles/GBuffer.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && C:/Strawberry/c/bin/c++.exe -shared -o ../../../bin/plugins/GBuffer.dll -Wl,--out-implib,libGBuffer.dll.a -Wl,--major-image-version,0,--minor-image-version,0 -Wl,--whole-archive CMakeFiles/GBuffer.dir/objects.a -Wl,--no-whole-archive @CMakeFiles/GBuffer.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/build: bin/plugins/GBuffer.dll
.PHONY : Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/build

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer && $(CMAKE_COMMAND) -P CMakeFiles/GBuffer.dir/cmake_clean.cmake
.PHONY : Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/clean

Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/DepthPass.3d.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferHelpers.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.cs.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.rt.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRT.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.cs.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.rt.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRT.slang
Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend: bin/shaders/RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/GBuffer C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/RenderPasses/GBuffer/CMakeFiles/GBuffer.dir/depend

