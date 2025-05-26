add_plugin(BiDirectionalPathTracer)

target_sources(BiDirectionalPathTracer PRIVATE
    BiDirectionalPathTracer.cpp
    BiDirectionalPathTracer.h
    CreateLightPaths.rt.slang
    CreateCameraPaths.rt.slang
    CombinePaths.rt.slang
    CollectBackProject.rt.slang
)

target_copy_shaders(BiDirectionalPathTracer RenderPasses/BiDirectionalPathTracer)
target_source_group(BiDirectionalPathTracer "RenderPasses")
