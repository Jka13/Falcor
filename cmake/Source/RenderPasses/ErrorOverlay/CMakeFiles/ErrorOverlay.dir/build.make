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
include Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/progress.make

# Include the compile flags for this target's objects.
include Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/flags.make

bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlay.cs.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cs.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "ErrorOverlay: Copying shader ErrorOverlay.cs.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cs.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlay.cs.slang

bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlayEnums.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlayEnums.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "ErrorOverlay: Copying shader ErrorOverlayEnums.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlayEnums.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlayEnums.slang

Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/flags.make
Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/includes_CXX.rsp
Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cpp
Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj -MF CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj.d -o CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cpp

Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cpp > CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.i

Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay/ErrorOverlay.cpp -o CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.s

# Object files for target ErrorOverlay
ErrorOverlay_OBJECTS = \
"CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj"

# External object files for target ErrorOverlay
ErrorOverlay_EXTERNAL_OBJECTS =

bin/plugins/ErrorOverlay.dll: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/ErrorOverlay.cpp.obj
bin/plugins/ErrorOverlay.dll: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/build.make
bin/plugins/ErrorOverlay.dll: Source/Falcor/libFalcor.dll.a
bin/plugins/ErrorOverlay.dll: external/fmt/libfmt.a
bin/plugins/ErrorOverlay.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/plugins/ErrorOverlay.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/plugins/ErrorOverlay.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/plugins/ErrorOverlay.dll: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/linkLibs.rsp
bin/plugins/ErrorOverlay.dll: Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX shared library ../../../bin/plugins/ErrorOverlay.dll"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/ErrorOverlay.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/ErrorOverlay.dir/objects.a @CMakeFiles/ErrorOverlay.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && C:/Strawberry/c/bin/c++.exe -shared -o ../../../bin/plugins/ErrorOverlay.dll -Wl,--out-implib,libErrorOverlay.dll.a -Wl,--major-image-version,0,--minor-image-version,0 -Wl,--whole-archive CMakeFiles/ErrorOverlay.dir/objects.a -Wl,--no-whole-archive @CMakeFiles/ErrorOverlay.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/build: bin/plugins/ErrorOverlay.dll
.PHONY : Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/build

Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay && $(CMAKE_COMMAND) -P CMakeFiles/ErrorOverlay.dir/cmake_clean.cmake
.PHONY : Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/clean

Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/depend: bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlay.cs.slang
Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/depend: bin/shaders/RenderPasses/ErrorOverlay/ErrorOverlayEnums.slang
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/ErrorOverlay C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/RenderPasses/ErrorOverlay/CMakeFiles/ErrorOverlay.dir/depend

