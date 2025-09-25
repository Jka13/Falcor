add_plugin(ComplexLuminaires)

target_sources(ComplexLuminaires PRIVATE
    ComplexLuminaires.cpp
    ComplexLuminaires.h
    Shader/GenerateSamples.rt.slang
    Shader/Debug.rt.slang
    Shader/DirectIllumination.cs.slang
    Shader/DirectIlluminationReference.rt.slang
    Shader/Helpers/SceneDataHandler.slang
    Shader/Helpers/ReservoirHandler.slang
    Shader/Sample.rt.slang
    Shader/Resample.cs.slang    
    Shader/Combine.cs.slang
)

target_copy_shaders(ComplexLuminaires RenderPasses/ComplexLuminaires)
target_source_group(ComplexLuminaires "RenderPasses")
