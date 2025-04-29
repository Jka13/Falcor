add_plugin(BiDirectionalPathTracer)

target_sources(BiDirectionalPathTracer PRIVATE
    BiDirectionalPathTracer.cpp
    BiDirectionalPathTracer.h
    CreatePaths.rt.slang
    CollectBackProject.rt.slang
)

target_source_group(BiDirectionalPathTracer "RenderPasses")
