import torch
from PIL import Image
from pathlib import Path
import sys

from py.cutter import Cutter
from py.utils import tensor_to_pil, mask_to_pil
sys.path.append(r"..\..")
import os

# Import the data creation script to ensure data exists
from tests.create_test_imgs import create_test_data, TEST_DATA_FOLDER

# --- Constants ---
TEST_OUTPUT_FOLDER = Path(__file__).parent.resolve() / "test_output"

# --- Main Test Execution ---

def run_cutter_test():
    """
    Loads test data, runs the Cutter.cut method, and saves the output.
    """
    print("--- Running Cutter Test ---")

    # 1. Ensure test data exists
    create_test_data()

    # 2. Load the test image and mask
    print("Loading test data...")
    image_path = TEST_DATA_FOLDER / "image_01.pt"
    mask_path = TEST_DATA_FOLDER / "mask_01.pt"

    if not image_path.exists() or not mask_path.exists():
        print("Error: Test data not found. Please run create_test_imgs.py first.")
        return

    test_image = torch.load(image_path)
    test_mask = torch.load(mask_path)
    print(f"Image loaded with shape: {test_image.shape}")
    print(f"Mask loaded with shape: {test_mask.shape}")

    # 3. Instantiate the node and execute its main function
    print("Instantiating Cutter and running the 'cut' method...")
    cutter_node = Cutter()
    cut_image, cut_mask = cutter_node.cut(image=test_image, opt_mask=test_mask)
    print(f"Result image shape: {cut_image.shape}")
    print(f"Result mask shape: {cut_mask.shape}")

    # 4. Save the results for visual inspection
    os.makedirs(TEST_OUTPUT_FOLDER, exist_ok=True)

    result_img_path = TEST_OUTPUT_FOLDER / "cutter_test_01_result_image.png"
    result_mask_path = TEST_OUTPUT_FOLDER / "cutter_test_01_result_mask.png"

    print(f"Saving result image to: {result_img_path}")
    tensor_to_pil(cut_image).save(result_img_path)

    print(f"Saving result mask to: {result_mask_path}")
    mask_to_pil(cut_mask).save(result_mask_path)

    print("--- Test Complete ---")


if __name__ == "__main__":
    run_cutter_test()
