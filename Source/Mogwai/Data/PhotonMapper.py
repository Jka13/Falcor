from falcor import *

def render_graph_RTPM():
    g = RenderGraph('RTPM')
    loadRenderPassLibrary('PhotonMapper.dll')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('BSDFViewer.dll')
    loadRenderPassLibrary('CompositeReStirVpl.dll')
    loadRenderPassLibrary('Antialiasing.dll')
    loadRenderPassLibrary('BlitPass.dll')
    loadRenderPassLibrary('ForwardLightingPass.dll')
    loadRenderPassLibrary('CompositeReStirNRD.dll')
    loadRenderPassLibrary('CSM.dll')
    loadRenderPassLibrary('DebugPasses.dll')
    loadRenderPassLibrary('PathTracer.dll')
    loadRenderPassLibrary('DepthPass.dll')
    loadRenderPassLibrary('DLSSPass.dll')
    loadRenderPassLibrary('ErrorMeasurePass.dll')
    loadRenderPassLibrary('SimplePostFX.dll')
    loadRenderPassLibrary('FLIPPass.dll')
    loadRenderPassLibrary('VBufferPM.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('PhotonReSTIRVPL.dll')
    loadRenderPassLibrary('ImageFilter.dll')
    loadRenderPassLibrary('WhittedRayTracer.dll')
    loadRenderPassLibrary('ImageLoader.dll')
    loadRenderPassLibrary('MegakernelPathTracer.dll')
    loadRenderPassLibrary('MinimalPathTracer.dll')
    loadRenderPassLibrary('ModulateIllumination.dll')
    loadRenderPassLibrary('NRDPass.dll')
    loadRenderPassLibrary('OptixDenoiser.dll')
    loadRenderPassLibrary('PassLibraryTemplate.dll')
    loadRenderPassLibrary('PhotonMapperHash.dll')
    loadRenderPassLibrary('PathVBuffer.dll')
    loadRenderPassLibrary('PhotonMapperStochasticHash.dll')
    loadRenderPassLibrary('PhotonReSTIR.dll')
    loadRenderPassLibrary('PhotonReStirFinalGathering.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    loadRenderPassLibrary('PTVBuffer.dll')
    loadRenderPassLibrary('ReStirExp.dll')
    loadRenderPassLibrary('SkyBox.dll')
    loadRenderPassLibrary('RTXDIPass.dll')
    loadRenderPassLibrary('RTXGIPass.dll')
    loadRenderPassLibrary('SceneDebugger.dll')
    loadRenderPassLibrary('SDFEditor.dll')
    loadRenderPassLibrary('SSAO.dll')
    loadRenderPassLibrary('SVGFPass.dll')
    loadRenderPassLibrary('TemporalDelayPass.dll')
    loadRenderPassLibrary('TestPasses.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    loadRenderPassLibrary('Utils.dll')
    loadRenderPassLibrary('Wireframe.dll')
    VBufferPM = createPass('VBufferPM', {'outputSize': IOSize.Default, 'samplePattern': 3, 'specRoughCutoff': 0.5, 'sampleCount': 32, 'useAlphaTest': True, 'adjustShadingNormals': True})
    g.addPass(VBufferPM, 'VBufferPM')
    RTPhotonMapper = createPass('RTPhotonMapper')
    g.addPass(RTPhotonMapper, 'RTPhotonMapper')
    ToneMapper = createPass('ToneMapper', {'outputSize': IOSize.Default, 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority})
    g.addPass(ToneMapper, 'ToneMapper')
    AccumulatePass = createPass('AccumulatePass', {'enabled': True, 'outputSize': IOSize.Default, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    g.addEdge('VBufferPM.vbuffer', 'RTPhotonMapper.vbuffer')
    g.addEdge('VBufferPM.throughput', 'RTPhotonMapper.thp')
    g.addEdge('VBufferPM.emissive', 'RTPhotonMapper.emissive')
    g.addEdge('VBufferPM.viewW', 'RTPhotonMapper.viewW')
    g.addEdge('RTPhotonMapper.PhotonImage', 'AccumulatePass.input')
    g.addEdge('AccumulatePass.output', 'ToneMapper.src')
    g.markOutput('ToneMapper.dst')
    g.markOutput('AccumulatePass.output')
    return g

RTPM = render_graph_RTPM()
try: m.addGraph(RTPM)
except NameError: None
