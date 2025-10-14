from math import ceil
import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image
import comfy
from .utils import tensor2pil, pil2tensor


# Define a dictionary of resampling filters
resample_filters = {"nearest": 0, "bilinear": 2, "bicubic": 3, "lanczos": 1}

# code from Interpause --->
# https://github.com/Interpause/auto-sd-paint-ext/blob/main/backend/utils.py


def sddebz_highres_fix(
    min_size: int,
    max_size: int,
    orig_width: int,
    orig_height: int,
    just_stride=False,
    stride=8,
):
    """Calculate an appropiate image resolution given the base input size of the
    model and max input size allowed.

    The max input size is due to how Stable Diffusion currently handles resolutions
    larger than its base/native input size of 512, which can cause weird issues
    such as duplicated features in the image. Hence, it is typically better to
    render at a smaller appropiate resolution before using other methods to upscale
    to the original resolution. Setting max_size to 512, matching the min_size,
    imitates how the highres fix works.

    Stable Diffusion also messes up for resolutions smaller than 512. In which case,
    it is better to render at the base resolution before downscaling to the original.

    This method requires less user input than the builtin highres fix, which uses
    firstphase_width and firstphase_height.

    The original plugin writer, @sddebz, wrote this. Interpause modified it to `ceil`
    instead of `round` to make selected region resizing easier in the plugin, and
    to avoid rounding to 0.

    Args:
        min_size (int): Native/base input size of the model.
        max_size (int): Max input size to accept.
        orig_width (int): Original width requested.
        orig_height (int): Original height requested.

    Returns:
        Tuple[int, int]: Appropiate (width, height) to use for the model.
    """

    def rnd(r, x):
        """Scale dimension x with stride z while attempting to preserve aspect ratio r."""
        return stride * ceil(r * x / stride)

    ratio = orig_width / orig_height

    # don't apply correction; just stride
    if just_stride:
        width, height = (
            ceil(orig_width / stride) * stride,
            ceil(orig_height / stride) * stride,
        )
    # height is smaller dimension
    elif orig_width > orig_height:
        width, height = rnd(ratio, min_size), min_size
        if width > max_size:
            width, height = max_size, rnd(1 / ratio, max_size)
    # width is smaller dimension
    else:
        width, height = min_size, rnd(1 / ratio, min_size)
        if height > max_size:
            width, height = rnd(ratio, max_size), max_size

    return width, height


# -->

# Implementation based on `https://github.com/lingondricka2/Upscaler-Detailer`

# code from comfyroll --->
# https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/blob/main/nodes/functions_upscale.py


def upscale_with_model(upscale_model, image):
    device = comfy.model_management.get_torch_device()
    upscale_model.to(device)
    # If the image has 3 dims, add batch_size:
    if image.ndim == 3:
        in_img = image.unsqueeze(0)
    else:
        in_img = image

    in_img = in_img.movedim(-1, -3).to(device)
    free_memory = comfy.model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                in_img.shape[2],
                in_img.shape[3],
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
            )
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(
                in_img,
                lambda a: upscale_model(a),
                tile_x=tile,
                tile_y=tile,
                overlap=overlap,
                upscale_amount=upscale_model.scale,
                pbar=pbar,
            )
            oom = False
        except comfy.model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
    # return 3 dims if the passed image had 3 dims
    return s if image.ndim != 3 else s.squeeze(0)


def apply_resize_image(
    image: Image.Image,
    original_width,
    original_height,
    rounding_modulus,
    mode="scale",
    supersample="true",
    factor: int = 2,
    width: int = 1024,
    height: int = 1024,
    resample="bicubic",
):
    if isinstance(image, torch.Tensor):
        from torchvision.transforms.functional import to_pil_image as tensor2pil
        from torchvision.transforms.functional import to_tensor as pil2tensor

        is_tensor = True
        image = tensor2pil(image)
    else:
        is_tensor = False

    # Calculate the new width and height based on the given mode and parameters
    if mode == "rescale":
        new_width, new_height = (
            int(original_width * factor),
            int(original_height * factor),
        )
    else:
        m = rounding_modulus
        original_ratio = original_height / original_width
        height = int(width * original_ratio)

        new_width = width if width % m == 0 else width + (m - width % m)
        new_height = height if height % m == 0 else height + (m - height % m)

    # Apply supersample
    if supersample == "true":
        image = image.resize(
            (new_width * 8, new_height * 8),
            resample=Image.Resampling(resample_filters[resample]),
        )

    # Resize the image using the given resampling filter
    resized_image = image.resize(
        (new_width, new_height), resample=Image.Resampling(resample_filters[resample])
    )

    if is_tensor:
        resized_image = pil2tensor(resized_image)

    return resized_image


def upscaler(
    image,
    upscale_model,
    rescale_factor,
    resampling_method,
    supersample,
    rounding_modulus,
):
    if upscale_model is not None:
        up_image = upscale_with_model(upscale_model, image)
    else:
        up_image = image

    pil_img = tensor2pil(image)
    original_width, original_height = pil_img.size
    scaled_image = pil2tensor(
        apply_resize_image(
            tensor2pil(up_image),
            original_width,
            original_height,
            rounding_modulus,
            "rescale",
            supersample,
            rescale_factor,
            1024,
            resampling_method,
        )
    )
    return scaled_image


# <---


def process_single_image(
    image, opt_mask, min_size, max_size, opt_upscale_model, scale_method
):
    orig_height, orig_width = image.shape[:2]  # Extract dimensions

    # Get the appropriate dimensions based on sddebz_highres_fix
    new_width, new_height = sddebz_highres_fix(
        min_size=min_size,
        max_size=max_size,
        orig_width=orig_width,
        orig_height=orig_height,
    )

    # Return the original if it's in bounds, though making sure it's divisible by 2
    if min_size <= orig_width <= max_size and min_size <= orig_height <= max_size:
        return image, opt_mask

    # Strictly use the upscale model only if the image actually needs an upscale
    if (orig_width < new_width or orig_height < new_height) and opt_upscale_model:
        resized_img = upscaler(
            image,
            opt_upscale_model,
            rescale_factor=new_width / orig_width,
            resampling_method=scale_method,
            supersample=False,
            rounding_modulus=2,
        )
    # Resize normally if we don't have an upscale model
    else:
        pil_img = tensor2pil(image)
        resized_img = apply_resize_image(
            pil_img,
            orig_width,
            orig_height,
            width=new_width,
            height=new_height,
            supersample=False,
            resample=scale_method,
            rounding_modulus=2,
        )
        # Convert back to tensor
        resized_img = pil2tensor(resized_img)

    resized_mask = None
    if opt_mask is not None:
        resized_mask = apply_resize_image(
            opt_mask,
            orig_width,
            orig_height,
            width=new_width,
            height=new_height,
            resample="nearest",
            supersample=False,
            rounding_modulus=2,
        )

    return resized_img, resized_mask


class Resize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"default": None}),
                "min_size": (
                    "INT",
                    {"default": 512, "min": 64, "max": 16384, "step": 64},
                ),
                "max_size": (
                    "INT",
                    {"default": 1536, "min": 128, "max": 16384, "step": 64},
                ),
                "scale_method": ([*resample_filters.keys()], {"default": "lanczos"}),
            },
            "optional": {
                "opt_upscale_model": (
                    "UPSCALE_MODEL",
                    {"default": None},
                ),  # Optional upscaling model
                "opt_mask": ("MASK", {"default": None}),  # Optional mask
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize_to_bounds"
    CATEGORY = "image"

    def resize_to_bounds(
        self,
        image,
        min_size,
        max_size,
        scale_method,
        opt_upscale_model=None,
        opt_mask=None,
    ):
        """
        Resizes an image (and optional mask) within given bounds.

        Args:
            img (tensor): The input image.
            min_size (int): Minimum allowed size for any dimension.
            max_size (int): Maximum allowed size for any dimension.
            upscale_model (Optional): An optional upscaling model (not used here).
            mask (Optional[PIL.Image.Image]): An optional mask to resize along with the image.

        Returns:
            Tuple[Image.Image, Optional[Image.Image]]: The resized image and mask.
        """
        processed_images = []
        processed_masks = []

        # Check if the image has a batch dimension
        if image.ndim == 4:  # Batch of images (B, H, W, C or B, C, H, W)
            batch_size = image.size(0)
            for i in range(batch_size):
                single_image = image[i]  # Extract the i-th image
                single_mask = opt_mask[i] if opt_mask is not None else None

                # Process a single image
                processed_img, processed_mask = process_single_image(
                    single_image,
                    single_mask,
                    min_size=min_size,
                    max_size=max_size,
                    opt_upscale_model=opt_upscale_model,
                    scale_method=scale_method,
                )
                processed_images.append(processed_img)
                if processed_mask is not None:
                    processed_masks.append(processed_mask)
        else:
            # No batch dimension, process as a single image
            processed_img, processed_mask = process_single_image(
                image,
                opt_mask,
                min_size=min_size,
                max_size=max_size,
                opt_upscale_model=opt_upscale_model,
                scale_method=scale_method,
            )
            processed_images.append(processed_img)
            if processed_mask is not None:
                processed_masks.append(processed_mask)

        # Return processed tensors
        return torch.stack(processed_images), torch.stack(
            processed_masks
        ) if processed_masks else None
