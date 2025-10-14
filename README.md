# poi-nodes
Custom nodes for my personal use. No guarantees for support / updates. Mainly intended for the context of inpainting or img2img (same thing). I prefer to use krita to visually test the outputs of the nodes, I would recommend anyone else to test their workflows in the same way.

## Changelog 2025-10-14
- Added example workflow `tests/poi-nodes.json`
- Renamed Smart Resize to Resize
- Added Crop node (behavior explained in lower section)
- Added Paste node (behavior explained in lower section)
- Replaced code in tests with an example workflow, faster dev with hot-reloading

## Resize (Renamed from Smart Resize)
**You should probably not use this node right now and instead use someone else's. It works but it's scuffed.**
This node will resize an image so that it is within the two specified sizes, optionally resizing a mask alongside it. It's possible to provide an upscale model rather than using the pre-defined algorithmic methods. Will not resize an image that is within the specified sizes. This behavior should be changed in the future to account for divisibility.

## Crop
This node will crop an image to the specified dimensions, optionally cropping a mask alongside it. Allows for expanding the context (black padding on the mask, more image content for the model). Also allows playing with that context to fit the nearest image that is divisible by a specified factor. You should use two crop nodes if you use these parameters. One that can add padding and one that can crop it again.

Given the case that a fully black or white mask is passed, the crop node will return a white mask matching the image size, allowing the reuse of workflows for inpainting and img2img.

## Paste
This node will paste an image onto another image based on mask coordinates, It's expected to be used in conjuction with the original mask that was fed to the first crop node (refer to example workflow).
