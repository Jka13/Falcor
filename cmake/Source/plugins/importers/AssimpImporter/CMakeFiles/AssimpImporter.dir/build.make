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
include Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/progress.make

# Include the compile flags for this target's objects.
include Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/flags.make

Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/flags.make
Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/includes_CXX.rsp
Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/plugins/importers/AssimpImporter/AssimpImporter.cpp
Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj -MF CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj.d -o CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/plugins/importers/AssimpImporter/AssimpImporter.cpp

Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/plugins/importers/AssimpImporter/AssimpImporter.cpp > CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.i

Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/plugins/importers/AssimpImporter/AssimpImporter.cpp -o CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.s

# Object files for target AssimpImporter
AssimpImporter_OBJECTS = \
"CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj"

# External object files for target AssimpImporter
AssimpImporter_EXTERNAL_OBJECTS =

bin/plugins/AssimpImporter.dll: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/AssimpImporter.cpp.obj
bin/plugins/AssimpImporter.dll: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/build.make
bin/plugins/AssimpImporter.dll: Source/Falcor/libFalcor.dll.a
bin/plugins/AssimpImporter.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/deps/lib/assimp-vc143-mt.lib
bin/plugins/AssimpImporter.dll: external/fmt/libfmt.a
bin/plugins/AssimpImporter.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/plugins/AssimpImporter.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/plugins/AssimpImporter.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/plugins/AssimpImporter.dll: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/linkLibs.rsp
bin/plugins/AssimpImporter.dll: Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library ../../../../bin/plugins/AssimpImporter.dll"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/AssimpImporter.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/AssimpImporter.dir/objects.a @CMakeFiles/AssimpImporter.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && C:/Strawberry/c/bin/c++.exe -shared -o ../../../../bin/plugins/AssimpImporter.dll -Wl,--out-implib,libAssimpImporter.dll.a -Wl,--major-image-version,0,--minor-image-version,0 -Wl,--whole-archive CMakeFiles/AssimpImporter.dir/objects.a -Wl,--no-whole-archive @CMakeFiles/AssimpImporter.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/build: bin/plugins/AssimpImporter.dll
.PHONY : Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/build

Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter && $(CMAKE_COMMAND) -P CMakeFiles/AssimpImporter.dir/cmake_clean.cmake
.PHONY : Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/clean

Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/plugins/importers/AssimpImporter C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/plugins/importers/AssimpImporter/CMakeFiles/AssimpImporter.dir/depend

