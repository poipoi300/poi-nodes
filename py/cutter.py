from math import ceil
import torch
import torch.nn.functional as F
import logging


class Cutter:
    def __init__(self):
        # A detached and logical perspective confirms the design of this node is to
        # crop an image and its corresponding mask based on the mask's active area.
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "opt_mask": ("MASK",),
                "mask_expand": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                # New parameters
                "ctx_expand": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "try_divisible_by": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 8},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image_cropped", "mask_cropped", "mask")
    FUNCTION = "process"
    CATEGORY = "image"

    @staticmethod
    def handle_mask(opt_mask=None):
        """
        Determines if a valid, non-trivial mask is provided.
        Expects a PyTorch Tensor with values in [0, 1].
        """
        if opt_mask is None:
            return False

        if opt_mask.numel() > 0 and (
            torch.all(opt_mask == 0.0) or torch.all(opt_mask == 1.0)
        ):
            return False

        coords = torch.argwhere(opt_mask)
        # Effectively empty mask (all zeros)
        if len(coords) == 0:
            return False

        return True

    @staticmethod
    def _process_mask(
        mask_tensor, height, width, y_min, x_min, y_max, x_max, mask_expand
    ):
        """
        Applies expansion to the mask, using only the relevant cropped section.
        Blur is now handled in the stitcher for better positioning accuracy.
        """
        cropped_mask = mask_tensor[y_min : y_max + 1, x_min : x_max + 1].clone()

        # We start with a full-size mask copy to place the scaled content back into
        processed_mask = mask_tensor.clone()

        # 1. Mask Expansion (Zoom)
        if mask_expand > 0.0:
            scale_factor = mask_expand + 1
            scaled_mask = (
                F.interpolate(
                    cropped_mask.unsqueeze(0).unsqueeze(0),
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )

            # Create a new full-size mask tensor for placement
            new_mask = torch.zeros_like(mask_tensor, dtype=torch.float32)

            sh, sw = scaled_mask.shape
            h, w = new_mask.shape

            # Center the scaled mask relative to the original image center
            original_center_y = (y_min + y_max) / 2.0
            original_center_x = (x_min + x_max) / 2.0

            start_y = int(original_center_y - sh / 2.0)
            start_x = int(original_center_x - sw / 2.0)

            # Clamp the coordinates to the boundaries of new_mask (full-sized)
            y_start = max(0, start_y)
            x_start = max(0, start_x)
            y_end = min(h, start_y + sh)
            x_end = min(w, start_x + sw)

            # Adjust the slice of scaled_mask in case of clamping
            scaled_y_start = 0 if start_y >= 0 else -start_y
            scaled_x_start = 0 if start_x >= 0 else -start_x
            scaled_y_end = sh if start_y + sh <= h else sh - ((start_y + sh) - h)
            scaled_x_end = sw if start_x + sw <= w else sw - ((start_x + sw) - w)

            new_mask[y_start:y_end, x_start:x_end] = scaled_mask[
                scaled_y_start:scaled_y_end, scaled_x_start:scaled_x_end
            ]

            processed_mask = new_mask

        return processed_mask

    @staticmethod
    def expand_bounds(height, width, y_min, x_min, y_max, x_max, ctx_expand):
        # Calculate border size as a fraction of the current crop's smallest dimension
        crop_h = y_max - y_min + 1
        crop_w = x_max - x_min + 1
        min_dim = min(crop_h, crop_w)

        # The border size in pixels, must be at least 1, and clamped by image edge
        border_size = ceil(min_dim * ctx_expand)

        # Expand bounds for image/mask to be cropped
        y_min = max(0, y_min - border_size)
        x_min = max(0, x_min - border_size)
        y_max = min(height - 1, y_max + border_size)
        x_max = min(width - 1, x_max + border_size)

        return y_min, x_min, y_max, x_max

    @staticmethod
    def try_adjust_bounds_for_divisibility(
        height,
        width,
        y_min,
        x_min,
        y_max,
        x_max,
        div,
        original_y_min,
        original_x_min,
        original_y_max,
        original_x_max,
    ):
        """
        Adjusts bounds to make the crop dimensions divisible by div.
        Prioritizes removing black borders first, then adds padding if needed.
        Only shrinks from areas that don't contain mask content (outside original mask bounds).
        Returns adjusted bounds that may not achieve perfect divisibility if constrained by image edges.
        """
        if div <= 0:
            return y_min, x_min, y_max, x_max

        current_h = y_max - y_min + 1
        current_w = x_max - x_min + 1

        print(f"DEBUG: Original bounds: ({y_min}, {x_min}) -> ({y_max}, {x_max})")
        print(f"DEBUG: Current dimensions: {current_h}x{current_w}, div={div}")

        # Calculate how much we need to adjust to be divisible
        h_remainder = current_h % div
        w_remainder = current_w % div

        print(f"DEBUG: Remainders: h={h_remainder}, w={w_remainder}")

        # For height adjustment
        if h_remainder != 0:
            # Calculate pixels needed for shrinking (to lower multiple) vs expanding (to higher multiple)
            shrink_needed = h_remainder  # pixels to remove to reach lower multiple
            expand_needed = div - h_remainder  # pixels to add to reach higher multiple
            print(
                f"DEBUG: Height shrink needs {shrink_needed}, expand needs {expand_needed} pixels"
            )

            # Try to remove black borders first (shrink crop) - but don't cut into mask content
            max_safe_shrink_top = (
                original_y_min - y_min
            )  # how much we can shrink from top without cutting mask
            max_safe_shrink_bottom = (
                y_max - original_y_max
            )  # how much we can shrink from bottom without cutting mask
            safe_shrink_top = min(shrink_needed // 2, y_min, max_safe_shrink_top)
            safe_shrink_bottom = min(
                shrink_needed - safe_shrink_top,
                height - 1 - y_max,
                max_safe_shrink_bottom,
            )
            total_safe_shrink = safe_shrink_top + safe_shrink_bottom
            print(
                f"DEBUG: Can safely shrink top={safe_shrink_top}, bottom={safe_shrink_bottom}, total={total_safe_shrink}"
            )

            if total_safe_shrink >= shrink_needed:
                # We can achieve divisibility by shrinking to lower multiple without cutting mask content
                y_min += safe_shrink_top
                y_max -= safe_shrink_bottom
                print(
                    f"DEBUG: Height achieved by safe shrinking: new bounds ({y_min}, {y_max})"
                )
            else:
                # Need to expand to higher multiple
                expand_top = min(expand_needed // 2, y_min)
                expand_bottom = min(expand_needed - expand_top, height - 1 - y_max)
                print(
                    f"DEBUG: Need to expand top={expand_top}, bottom={expand_bottom} (needed={expand_needed})"
                )

                y_min -= expand_top
                y_max += expand_bottom
                print(f"DEBUG: Height after expansion: new bounds ({y_min}, {y_max})")

        # For width adjustment
        if w_remainder != 0:
            # Calculate pixels needed for shrinking (to lower multiple) vs expanding (to higher multiple)
            shrink_needed = w_remainder  # pixels to remove to reach lower multiple
            expand_needed = div - w_remainder  # pixels to add to reach higher multiple
            print(
                f"DEBUG: Width shrink needs {shrink_needed}, expand needs {expand_needed} pixels"
            )

            # Try to remove black borders first (shrink crop) - but don't cut into mask content
            max_safe_shrink_left = (
                original_x_min - x_min
            )  # how much we can shrink from left without cutting mask
            max_safe_shrink_right = (
                x_max - original_x_max
            )  # how much we can shrink from right without cutting mask
            safe_shrink_left = min(shrink_needed // 2, x_min, max_safe_shrink_left)
            safe_shrink_right = min(
                shrink_needed - safe_shrink_left,
                width - 1 - x_max,
                max_safe_shrink_right,
            )
            total_safe_shrink = safe_shrink_left + safe_shrink_right
            print(
                f"DEBUG: Can safely shrink left={safe_shrink_left}, right={safe_shrink_right}, total={total_safe_shrink}"
            )

            if total_safe_shrink >= shrink_needed:
                # We can achieve divisibility by shrinking to lower multiple without cutting mask content
                x_min += safe_shrink_left
                x_max -= safe_shrink_right
                print(
                    f"DEBUG: Width achieved by safe shrinking: new bounds ({x_min}, {x_max})"
                )
            else:
                # Need to expand to higher multiple
                expand_left = min(expand_needed // 2, x_min)
                expand_right = min(expand_needed - expand_left, width - 1 - x_max)
                print(
                    f"DEBUG: Need to expand left={expand_left}, right={expand_right} (needed={expand_needed})"
                )

                x_min -= expand_left
                x_max += expand_right
                print(f"DEBUG: Width after expansion: new bounds ({x_min}, {x_max})")

        final_h = y_max - y_min + 1
        final_w = x_max - x_min + 1
        print(f"DEBUG: Final bounds: ({y_min}, {x_min}) -> ({y_max}, {x_max})")
        print(
            f"DEBUG: Final dimensions: {final_h}x{final_w}, h%{div}={final_h%div}, w%{div}={final_w%div}"
        )

        return y_min, x_min, y_max, x_max

    def process(
        self, image, opt_mask=None, mask_expand=1.0, ctx_expand=0.0, try_divisible_by=0
    ):
        logger = logging.getLogger(__name__)
        device = image.device
        batch_size, height, width, _ = image.shape
        if opt_mask is not None:
            print(f"DEBUG: Batch mask shape: {opt_mask.shape}")

        def process_single(batch_idx):
            single_image = image[batch_idx].squeeze()
            single_mask = (
                opt_mask[batch_idx].squeeze() if opt_mask is not None else None
            )

            # 1. Handle Input Trivial Mask
            if not self.handle_mask(single_mask):
                logger.warning("No mask provided, using white mask")
                white_mask = torch.ones(
                    1, height, width, dtype=torch.float32, device=device
                )
                return (single_image, white_mask, white_mask)

            logger.info(f"Processing image with shape {image.shape}")
            logger.info(f"Mask shape: {single_mask.shape}")  # 2D
            if single_mask.shape != (height, width):
                raise ValueError(
                    f"Mask dimensions ({single_mask.shape}) do not match image dimensions ({image.shape})."
                )

            coords = torch.argwhere(single_mask)
            logger.info(f"Mask coordinates: {coords}")

            y_min, x_min = torch.min(coords, dim=0).values
            y_max, x_max = torch.max(coords, dim=0).values
            logger.info(f"Mask bounds: ({y_min}, {x_min}) -> ({y_max}, {x_max})")

            # 1. Perform Mask Processing (Expansion only) - blur moved to stitcher
            processed_mask_tensor = self._process_mask(
                single_mask, height, width, y_min, x_min, y_max, x_max, mask_expand
            )

            # Save the unmodified original mask for accurate positioning in stitcher
            full_unmodified_mask = single_mask.clone()

            # 2. Apply Context Expansion (Border)
            if ctx_expand > 0.0:
                y_min, x_min, y_max, x_max = self.expand_bounds(
                    height, width, y_min, x_min, y_max, x_max, ctx_expand
                )

            # Store original mask bounds before any modifications
            original_coords = torch.argwhere(single_mask)
            original_y_min, original_x_min = torch.min(original_coords, dim=0).values
            original_y_max, original_x_max = torch.max(original_coords, dim=0).values

            # 3. Enforce Divisibility (adjust bounds for divisibility)
            if try_divisible_by > 1:
                y_min, x_min, y_max, x_max = self.try_adjust_bounds_for_divisibility(
                    height,
                    width,
                    y_min,
                    x_min,
                    y_max,
                    x_max,
                    try_divisible_by,
                    original_y_min,
                    original_x_min,
                    original_y_max,
                    original_x_max,
                )

            # The final cropped image area is defined by the full extent of all modifications
            cropped_image_tensor = single_image[y_min : y_max + 1, x_min : x_max + 1, :]

            # Crop the mask using the final bounds
            final_cropped_mask_tensor = processed_mask_tensor[
                y_min : y_max + 1, x_min : x_max + 1
            ]
            # Add the batch back in after lost from indexing
            final_cropped_mask_tensor_batched = final_cropped_mask_tensor.to(device)

            # Full unmodified mask for accurate positioning in stitcher
            full_unmodified_mask_batched = full_unmodified_mask

            return (
                cropped_image_tensor,
                final_cropped_mask_tensor_batched,
                full_unmodified_mask_batched,
            )

        results = [process_single(i) for i in range(batch_size)]
        images, cropped_masks, full_masks = zip(*results)
        return (
            torch.stack(images),
            torch.stack(cropped_masks),
            torch.stack(full_masks),
        )
