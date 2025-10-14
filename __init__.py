# Nodes
from .py.crop import Crop
from .py.resize import Resize
from .py.paste import Paste

NODE_CLASS_MAPPINGS = {"Crop": Crop, "Resize": Resize, "Paste": Paste}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Crop": "Crop",
    "Resize": "Resize",
    "Paste": "Paste",
}
