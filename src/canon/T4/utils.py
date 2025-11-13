from canon.config import BASE_DATA_PATH
from pathlib import Path
import cv2

def load_image_and_masks(name: str):
    base = BASE_DATA_PATH / "T4"
    print("Reading from", base)
    img_path = base / "imagens" / f"{name}.jpg"
    if not img_path.exists():
        img_path = base / "imagens" / f"{name}.png"

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    masks = []
    for i in range(1, 4):
        mask_path = base / "mascaras" / f"{name}_mask_{i}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masks.append(mask)

    return image, masks


def apply_mask(image, mask):
    # Invert mask so white (to remove) becomes 0, black (keep) becomes 255
    inverse_mask = cv2.bitwise_not(mask)

    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=inverse_mask)
    return result