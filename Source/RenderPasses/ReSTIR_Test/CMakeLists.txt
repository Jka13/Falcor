add_plugin(ReSTIR_Test)

target_sources(ReSTIR_Test PRIVATE
    ReSTIR_Test.cpp
    ReSTIR_Test.h
    Shader/GenerateSamples.rt.slang
    Shader/Resample.cs.slang
    Shader/Combine.cs.slang
    Shader/Helpers/SceneDataHandler.slang
    Shader/Helpers/ReservoirHandler.slang
)

target_copy_shaders(ReSTIR_Test RenderPasses/ReSTIR_Test)
target_source_group(ReSTIR_Test "RenderPasses")
