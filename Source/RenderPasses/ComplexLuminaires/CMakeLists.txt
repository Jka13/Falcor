add_plugin(ComplexLuminaires)

target_sources(ComplexLuminaires PRIVATE
    ComplexLuminaires.cpp
    ComplexLuminaires.h
    Shader/GenerateSamples.rt.slang
    Shader/Helpers/SceneDataHandler.slang
    Shader/Helpers/ReservoirHandler.slang
)

target_copy_shaders(ComplexLuminaires RenderPasses/ComplexLuminaires)
target_source_group(ComplexLuminaires "RenderPasses")
