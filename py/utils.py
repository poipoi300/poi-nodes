import torch
from torchvision.transforms import ToPILImage, ToTensor
from PIL import Image


def tensor2pil(tensor):
    # Check if the tensor has 3, 4 or 5 dimensions
    if tensor.ndim == 4:  # Batch of images (B, H, W, C)
        tensor = tensor.permute(0, 3, 1, 2).squeeze()  # Rearrange to (B, C, H, W)
    elif tensor.ndim == 3:  # Single image (H, W, C)
        tensor = tensor.permute(2, 0, 1)  # Rearrange to (C, H, W)
    return ToPILImage()(tensor)

def pil2tensor(pil_image):
    tensor = ToTensor()(pil_image)  # Produces (C, H, W)
    return tensor.permute(1, 2, 0)  # Rearrange to (H, W, C)

def mask_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a ComfyUI-style mask tensor (B, H, W) to a PIL Image."""
    # Take the first mask in the batch
    mask_tensor = tensor[0]
    # Scale from [0, 1] to [0, 255]
    mask_tensor_8bit = mask_tensor.mul(255).byte()
    # Convert to NumPy array and then to a grayscale PIL Image
    return Image.fromarray(mask_tensor_8bit.cpu().numpy(), 'L')
