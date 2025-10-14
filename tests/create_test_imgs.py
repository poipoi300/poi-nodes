import torch
from pathlib import Path
import os

# Define the directory for test assets
TEST_DATA_FOLDER = Path(__file__).parent.resolve() / "test_data"

# --- Main data creation function ---

def create_test_data():
    """
    Generates and saves a sample image and mask for testing if they do not already exist.
    The output format matches ComfyUI's standard:
    - Image: torch.Tensor, shape (B, H, W, C), dtype=float32, range [0, 1]
    - Mask:  torch.Tensor, shape (B, H, W), dtype=float32, range [0, 1]
    """
    IMG_H, IMG_W = 512, 768
    BBOX_X, BBOX_Y = 192, 128
    BBOX_W, BBOX_H = 384, 256

    img_path = TEST_DATA_FOLDER / "image_01.pt"
    mask_path = TEST_DATA_FOLDER / "mask_01.pt"

    # Create the directory if it does not exist
    os.makedirs(TEST_DATA_FOLDER, exist_ok=True)

    # --- Image Generation ---
    if not img_path.exists():
        print(f"Generating test image: {img_path}")
        # Create a blue background (Batch=1, H, W, Channels=3)
        image_tensor = torch.zeros(1, IMG_H, IMG_W, 3, dtype=torch.float32)
        image_tensor[:, :, :, 2] = 0.8  # Blue channel

        # Add a red rectangle in a specific region
        image_tensor[:, BBOX_Y:(BBOX_Y + BBOX_H), BBOX_X:(BBOX_X + BBOX_W), 0] = 0.9  # Red channel
        image_tensor[:, BBOX_Y:(BBOX_Y + BBOX_H), BBOX_X:(BBOX_X + BBOX_W), 2] = 0.0   # Clear blue channel

        torch.save(image_tensor, img_path)
    else:
        print(f"Test image already exists: {img_path}")

    # --- Mask Generation ---
    if not mask_path.exists():
        print(f"Generating test mask: {mask_path}")
        # Create a black mask (Batch=1, H, W)
        mask_tensor = torch.zeros(1, IMG_H, IMG_W, dtype=torch.float32)

        # Add a white rectangle corresponding to the image's red box
        mask_tensor[:, BBOX_Y:(BBOX_Y + BBOX_H), BBOX_X:(BBOX_X + BBOX_W)] = 1.0

        torch.save(mask_tensor, mask_path)
    else:
        print(f"Test mask already exists: {mask_path}")

if __name__ == "__main__":
    create_test_data()
