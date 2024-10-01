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
include Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/progress.make

# Include the compile flags for this target's objects.
include Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/flags.make

bin/shaders/Samples/MultiSampling/MultiSampling.3d.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.3d.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "MultiSampling: Copying shader MultiSampling.3d.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.3d.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/Samples/MultiSampling/MultiSampling.3d.slang

Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/flags.make
Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/includes_CXX.rsp
Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.cpp
Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj -MF CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj.d -o CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.cpp

Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/MultiSampling.dir/MultiSampling.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.cpp > CMakeFiles/MultiSampling.dir/MultiSampling.cpp.i

Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/MultiSampling.dir/MultiSampling.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling/MultiSampling.cpp -o CMakeFiles/MultiSampling.dir/MultiSampling.cpp.s

# Object files for target MultiSampling
MultiSampling_OBJECTS = \
"CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj"

# External object files for target MultiSampling
MultiSampling_EXTERNAL_OBJECTS =

bin/MultiSampling.exe: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/MultiSampling.cpp.obj
bin/MultiSampling.exe: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/build.make
bin/MultiSampling.exe: Source/Falcor/libFalcor.dll.a
bin/MultiSampling.exe: external/fmt/libfmt.a
bin/MultiSampling.exe: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/MultiSampling.exe: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/MultiSampling.exe: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/MultiSampling.exe: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/linkLibs.rsp
bin/MultiSampling.exe: Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../../../bin/MultiSampling.exe"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/MultiSampling.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/MultiSampling.dir/objects.a @CMakeFiles/MultiSampling.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && C:/Strawberry/c/bin/c++.exe -Wl,--whole-archive CMakeFiles/MultiSampling.dir/objects.a -Wl,--no-whole-archive -o ../../../bin/MultiSampling.exe -Wl,--out-implib,libMultiSampling.dll.a -Wl,--major-image-version,0,--minor-image-version,0 @CMakeFiles/MultiSampling.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/build: bin/MultiSampling.exe
.PHONY : Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/build

Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling && $(CMAKE_COMMAND) -P CMakeFiles/MultiSampling.dir/cmake_clean.cmake
.PHONY : Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/clean

Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/depend: bin/shaders/Samples/MultiSampling/MultiSampling.3d.slang
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/Samples/MultiSampling C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/Samples/MultiSampling/CMakeFiles/MultiSampling.dir/depend

