from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_g():
    g = RenderGraph('ComplexLuminaires')
    g.create_pass('ComplexLuminaires', 'ComplexLuminaires', {})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'cullNonOpaque': False, 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('ComplexLuminaires.color', 'AccumulatePass.input')
    g.add_edge('VBufferRT.vbuffer', 'ComplexLuminaires.vBuffer')
    g.add_edge('VBufferRT.viewW', 'ComplexLuminaires.view')
    g.add_edge('VBufferRT.mvec', 'ComplexLuminaires.motionVector')
    g.mark_output('AccumulatePass.output')
    g.mark_output('ToneMapper.dst')
    return g

g = render_graph_g()
try: m.addGraph(g)
except NameError: None
