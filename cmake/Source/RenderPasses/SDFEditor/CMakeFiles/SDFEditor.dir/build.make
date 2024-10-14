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
include Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/compiler_depend.make

# Include the progress variables for this target.
include Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/progress.make

# Include the compile flags for this target's objects.
include Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/flags.make

bin/shaders/RenderPasses/SDFEditor/GUIPass.ps.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/GUIPass.ps.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "SDFEditor: Copying shader GUIPass.ps.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/GUIPass.ps.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/SDFEditor/GUIPass.ps.slang

bin/shaders/RenderPasses/SDFEditor/Marker2DSet.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "SDFEditor: Copying shader Marker2DSet.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/SDFEditor/Marker2DSet.slang

bin/shaders/RenderPasses/SDFEditor/Marker2DTypes.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DTypes.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "SDFEditor: Copying shader Marker2DTypes.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DTypes.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/SDFEditor/Marker2DTypes.slang

bin/shaders/RenderPasses/SDFEditor/SDFEditorTypes.slang: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditorTypes.slang
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "SDFEditor: Copying shader SDFEditorTypes.slang"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && "C:/Program Files/CMake/bin/cmake.exe" -E copy_if_different C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditorTypes.slang C:/Users/jonas/Documents/Falcor/Falcor/cmake/bin/shaders/RenderPasses/SDFEditor/SDFEditorTypes.slang

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/flags.make
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/includes_CXX.rsp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.cpp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj -MF CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj.d -o CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.cpp

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.cpp > CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.i

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/Marker2DSet.cpp -o CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.s

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/flags.make
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/includes_CXX.rsp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditor.cpp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj -MF CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj.d -o CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditor.cpp

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/SDFEditor.dir/SDFEditor.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditor.cpp > CMakeFiles/SDFEditor.dir/SDFEditor.cpp.i

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/SDFEditor.dir/SDFEditor.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SDFEditor.cpp -o CMakeFiles/SDFEditor.dir/SDFEditor.cpp.s

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/flags.make
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/includes_CXX.rsp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj: C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SelectionWheel.cpp
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj -MF CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj.d -o CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj -c C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SelectionWheel.cpp

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.i"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SelectionWheel.cpp > CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.i

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.s"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor/SelectionWheel.cpp -o CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.s

# Object files for target SDFEditor
SDFEditor_OBJECTS = \
"CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj" \
"CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj" \
"CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj"

# External object files for target SDFEditor
SDFEditor_EXTERNAL_OBJECTS =

bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/Marker2DSet.cpp.obj
bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SDFEditor.cpp.obj
bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/SelectionWheel.cpp.obj
bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/build.make
bin/plugins/SDFEditor.dll: Source/Falcor/libFalcor.dll.a
bin/plugins/SDFEditor.dll: external/fmt/libfmt.a
bin/plugins/SDFEditor.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/python/libs/python310.lib
bin/plugins/SDFEditor.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/slang.lib
bin/plugins/SDFEditor.dll: C:/Users/jonas/Documents/Falcor/Falcor/external/packman/slang/bin/windows-x64/release/gfx.lib
bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/linkLibs.rsp
bin/plugins/SDFEditor.dll: Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=C:/Users/jonas/Documents/Falcor/Falcor/cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared library ../../../bin/plugins/SDFEditor.dll"
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && "C:/Program Files/CMake/bin/cmake.exe" -E rm -f CMakeFiles/SDFEditor.dir/objects.a
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/ar.exe qc CMakeFiles/SDFEditor.dir/objects.a @CMakeFiles/SDFEditor.dir/objects1.rsp
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && C:/Strawberry/c/bin/c++.exe -shared -o ../../../bin/plugins/SDFEditor.dll -Wl,--out-implib,libSDFEditor.dll.a -Wl,--major-image-version,0,--minor-image-version,0 -Wl,--whole-archive CMakeFiles/SDFEditor.dir/objects.a -Wl,--no-whole-archive @CMakeFiles/SDFEditor.dir/linkLibs.rsp

# Rule to build all files generated by this target.
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/build: bin/plugins/SDFEditor.dll
.PHONY : Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/build

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/clean:
	cd C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor && $(CMAKE_COMMAND) -P CMakeFiles/SDFEditor.dir/cmake_clean.cmake
.PHONY : Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/clean

Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend: bin/shaders/RenderPasses/SDFEditor/GUIPass.ps.slang
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend: bin/shaders/RenderPasses/SDFEditor/Marker2DSet.slang
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend: bin/shaders/RenderPasses/SDFEditor/Marker2DTypes.slang
Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend: bin/shaders/RenderPasses/SDFEditor/SDFEditorTypes.slang
	$(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" C:/Users/jonas/Documents/Falcor/Falcor C:/Users/jonas/Documents/Falcor/Falcor/Source/RenderPasses/SDFEditor C:/Users/jonas/Documents/Falcor/Falcor/cmake C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor C:/Users/jonas/Documents/Falcor/Falcor/cmake/Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : Source/RenderPasses/SDFEditor/CMakeFiles/SDFEditor.dir/depend

