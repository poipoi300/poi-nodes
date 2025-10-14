# Nodes
from .py.cutter import Cutter
from .py.resize import SmartResize
from .py.stitcher import Stitcher

NODE_CLASS_MAPPINGS = {
    "Cutter": Cutter,
    "SmartResize": SmartResize,
    "Stitcher": Stitcher
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Cutter": "Cutter",
    "SmartResize": "Smart Resize",
    "Stitcher": "Stitcher"
}
