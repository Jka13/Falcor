add_plugin(ComplexLuminairesReSTIR_PT)

target_sources(ComplexLuminairesReSTIR_PT PRIVATE
    ComplexLuminairesReSTIR_PT.cpp
    ComplexLuminairesReSTIR_PT.h
    Shader/GenerateSamples.rt.slang
    Shader/PathTracing.rt.slang
    Shader/PathResample.rt.slang
    Shader/Debug.rt.slang
    Shader/DirectIllumination.cs.slang
    Shader/DirectIlluminationReference.rt.slang
    Shader/Sample.rt.slang
    Shader/Resample.cs.slang    
    Shader/SplattingResample.cs.slang
    Shader/Combine.cs.slang
    Shader/SplattingCombine.cs.slang
    Shader/SortSplatReservoirs.cs.slang
    Shader/TemporalSplatReservoir.cs.slang
    Shader/Helpers/SceneDataHandler.slang
    Shader/Helpers/ReservoirHandler.slang
    Shader/Helpers/PathReservoirHandler.slang
    Shader/Helpers/VPLSampler.slang
)

target_copy_shaders(ComplexLuminairesReSTIR_PT RenderPasses/ComplexLuminairesReSTIR_PT)
target_source_group(ComplexLuminairesReSTIR_PT "RenderPasses")
