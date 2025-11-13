import torch
import torch.nn.functional as F
from distmap import euclidean_distance_transform


class Paste:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dest": ("IMAGE",),
                "image_source": ("IMAGE",),
                "output_mode": (["Transparent Background", "Destination Image"],),
            },
            "optional": {
                "opt_mask": ("MASK",),
                "blend": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "paste"
    CATEGORY = "image"

    def _gaussian_blur_2d(self, tensor, kernel_size, sigma):
        """Apply 2D Gaussian blur to a tensor using torch operations."""
        # Ensure kernel size is odd and reasonable
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, kernel_size)

        # Create 1D Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
        x = x - kernel_size // 2

        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create 2D kernel by outer product
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

        # Apply padding to maintain size
        padding = kernel_size // 2

        # Add batch and channel dimensions if needed
        original_shape = tensor.shape
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)

        # Apply convolution
        blurred = F.conv2d(tensor, kernel_2d, padding=padding)

        # Restore original shape
        if len(original_shape) == 2:
            blurred = blurred.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            blurred = blurred.squeeze(0)

        return blurred

    def _apply_inset_blend(
        self,
        mask,
        blend_distance,
        is_top_edge=False,
        is_left_edge=False,
        is_bottom_edge=False,
        is_right_edge=False,
    ):
        """Apply inset blend effect using euclidean distance transform, skipping edges that align with the canvas."""
        # Convert mask to binary (0 or 1)
        binary_mask = (mask > 0.5).float()

        # Pad the mask to ensure proper edge detection for the distance transform.
        # For edges that align with the canvas, we pad with 1s (inside the mask) to prevent blending.
        # For non-aligned edges, we pad with 0s (outside the mask) to create the blend.
        pad_size = int(blend_distance) + 2

        # Start by padding with 0s on all sides, which creates the blend effect.
        padded_mask = F.pad(
            binary_mask,
            (pad_size, pad_size, pad_size, pad_size),
            mode="constant",
            value=0,
        )

        # Overwrite padding with 1s on aligned edges to prevent blending there.
        if is_top_edge:
            padded_mask[:pad_size, :] = 1
        if is_bottom_edge:
            padded_mask[-pad_size:, :] = 1
        if is_left_edge:
            padded_mask[:, :pad_size] = 1
        if is_right_edge:
            padded_mask[:, -pad_size:] = 1

        # Apply euclidean distance transform to get distance from background (0s) for each foreground pixel
        # This gives us distance from edge inward for mask pixels
        distance_map_padded = euclidean_distance_transform(padded_mask)

        # Remove padding to get back to original size
        distance_map = distance_map_padded[pad_size:-pad_size, pad_size:-pad_size]

        # Create inset falloff: pixels close to edge (small distance) get low opacity
        # Pixels far from edge (large distance) get full opacity
        falloff = torch.clamp(distance_map / blend_distance, 0.0, 1.0)

        # Apply falloff only to original mask pixels
        result = binary_mask * falloff

        # Apply Gaussian blur to smooth out harsh transitions
        # This eliminates visible "rings" from discrete distance values
        blur_sigma = max(1.0, blend_distance * 0.15)
        kernel_size = max(3, int(blur_sigma * 3))
        if kernel_size % 2 == 0:
            kernel_size += 1

        result_smoothed = self._gaussian_blur_2d(result, kernel_size, blur_sigma)

        return result_smoothed

    def _resize_tensor(self, tensor, target_size):
        """Resize a tensor to target size using bilinear interpolation."""
        # tensor should be (H, W, C)
        if len(tensor.shape) == 3:
            # Convert to (1, C, H, W) for F.interpolate
            tensor_resized = tensor.permute(2, 0, 1).unsqueeze(0)
            tensor_resized = F.interpolate(
                tensor_resized, size=target_size, mode="bilinear", align_corners=False
            )
            # Convert back to (H, W, C)
            tensor_resized = tensor_resized.squeeze(0).permute(1, 2, 0)
        else:
            # Handle 2D tensors (masks)
            tensor_resized = tensor.unsqueeze(0).unsqueeze(0)
            tensor_resized = F.interpolate(
                tensor_resized, size=target_size, mode="bilinear", align_corners=False
            )
            tensor_resized = tensor_resized.squeeze(0).squeeze(0)

        return tensor_resized

    def _paste_with_mask(
        self, dest_tensor, source_tensor, mask_tensor, x_offset, y_offset
    ):
        """Paste source tensor onto destination tensor using mask for blending."""
        dest_height, dest_width = dest_tensor.shape[:2]
        source_height, source_width = source_tensor.shape[:2]

        # Calculate valid region bounds
        x_start = max(0, x_offset)
        y_start = max(0, y_offset)
        x_end = min(dest_width, x_offset + source_width)
        y_end = min(dest_height, y_offset + source_height)

        # Calculate source region bounds
        src_x_start = x_start - x_offset
        src_y_start = y_start - y_offset
        src_x_end = src_x_start + (x_end - x_start)
        src_y_end = src_y_start + (y_end - y_start)

        if x_start >= x_end or y_start >= y_end:
            return dest_tensor

        # Extract regions
        dest_region = dest_tensor[y_start:y_end, x_start:x_end]
        source_region = source_tensor[src_y_start:src_y_end, src_x_start:src_x_end]
        mask_region = mask_tensor[src_y_start:src_y_end, src_x_start:src_x_end]

        # Ensure mask has the right dimensions for broadcasting
        if len(dest_region.shape) == 3:  # Color image
            mask_region = mask_region.unsqueeze(-1)

        # Alpha blend: result = source * mask + dest * (1 - mask)
        blended_region = source_region * mask_region + dest_region * (1 - mask_region)

        # Create result tensor
        result = dest_tensor.clone()
        result[y_start:y_end, x_start:x_end] = blended_region

        return result

    def paste(self, image_dest, image_source, output_mode, opt_mask=None, blend=0.0):
        batch_size, height, width, channels = image_dest.shape

        if opt_mask is None or torch.all(opt_mask == 0.0) or torch.all(opt_mask == 1.0):
            return (image_dest,)

        if opt_mask.shape[1] != height or opt_mask.shape[2] != width:
            raise ValueError(
                f"Mask dimensions do not match image dimensions. Expected shape: ({batch_size}, {height}, {width}), got shape: {opt_mask.shape}"
            )

        # Work with first batch item
        dest_img = image_dest[0]  # (H, W, C)
        source_img = image_source[0]  # (H, W, C)
        mask = opt_mask[0]  # (H, W)

        # Find bounding box of the mask
        coords = torch.nonzero(mask)
        if coords.numel() == 0:
            return (image_dest,)

        y_min, x_min = torch.min(coords, dim=0).values
        y_max, x_max = torch.max(coords, dim=0).values

        bbox_height = y_max - y_min + 1
        bbox_width = x_max - x_min + 1

        # Resize source image to fit bounding box
        source_resized = self._resize_tensor(source_img, (bbox_height, bbox_width))

        # Crop mask to bounding box
        mask_cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

        # Apply blend effect if specified
        if blend > 0.0:
            # Calculate blend distance based on blend parameter
            min_dimension = min(bbox_height, bbox_width)
            blend_distance = blend * min_dimension * 0.5  # Scale with mask size
            blend_distance = max(blend_distance, 2.0)  # Ensure minimum blend

            # Determine which edges of the mask are aligned with the canvas edges
            is_top_edge = y_min == 0
            is_left_edge = x_min == 0
            is_bottom_edge = y_max == height - 1
            is_right_edge = x_max == width - 1

            # Apply inset blend using euclidean distance transform
            mask_float = mask_cropped.float()
            mask_cropped = self._apply_inset_blend(
                mask_float,
                blend_distance,
                is_top_edge=is_top_edge,
                is_left_edge=is_left_edge,
                is_bottom_edge=is_bottom_edge,
                is_right_edge=is_right_edge,
            )

        # Clamp mask values
        mask_cropped = torch.clamp(mask_cropped, 0.0, 1.0)

        if output_mode == "Transparent Background":
            # Create transparent background (RGBA)
            if channels == 3:
                # Convert RGB to RGBA
                dest_rgba = torch.zeros(
                    (height, width, 4), dtype=dest_img.dtype, device=dest_img.device
                )
            else:
                dest_rgba = torch.zeros_like(dest_img)
                if dest_rgba.shape[2] < 4:
                    # Pad to RGBA if needed
                    padding = torch.zeros(
                        (height, width, 4 - dest_rgba.shape[2]),
                        dtype=dest_img.dtype,
                        device=dest_img.device,
                    )
                    dest_rgba = torch.cat([dest_rgba, padding], dim=2)

            # Ensure source is RGBA
            if source_resized.shape[2] == 3:
                # Add alpha channel
                alpha = torch.ones(
                    int(bbox_height),
                    int(bbox_width),
                    1,
                    dtype=source_resized.dtype,
                    device=source_resized.device,
                )
                source_resized = torch.cat([source_resized, alpha], dim=2)

            # Set alpha channel based on mask for source
            source_resized = source_resized.clone()
            source_resized[..., 3] = mask_cropped

            # Create a dummy mask of all ones, since the alpha is already in the source image
            paste_mask = torch.ones_like(mask_cropped)

            # Paste using the dummy mask to avoid double-multiplying the alpha
            result = self._paste_with_mask(
                dest_rgba, source_resized, paste_mask, x_min.item(), y_min.item()
            )

        else:  # "Destination Image"
            # Ensure both images have same number of channels
            if dest_img.shape[2] != source_resized.shape[2]:
                if dest_img.shape[2] == 4 and source_resized.shape[2] == 3:
                    # Add alpha to source
                    alpha = torch.ones(
                        int(bbox_height),
                        int(bbox_width),
                        1,
                        dtype=source_resized.dtype,
                        device=source_resized.device,
                    )
                    source_resized = torch.cat([source_resized, alpha], dim=2)
                elif dest_img.shape[2] == 3 and source_resized.shape[2] == 4:
                    # Remove alpha from source
                    source_resized = source_resized[..., :3]

            # Paste with mask blending
            result = self._paste_with_mask(
                dest_img, source_resized, mask_cropped, x_min.item(), y_min.item()
            )

        return (result.unsqueeze(0),)
