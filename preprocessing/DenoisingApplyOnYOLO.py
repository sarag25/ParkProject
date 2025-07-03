import cv2
from pathlib import Path
from tqdm import tqdm
from SingleDenoiseFunctions import increase_contrast, denoise_median

'''
Function used in preprocessing to apply denoising to the raw dataset for object detection 
and then save the result as dataset_yolo
'''

src_root = Path("...")  # raw dataset before denoising
dst_root = Path("../datasets/dataset_yolo") # dataset after denoising


def process_split(split):
    """
    Apply the procedure to all 3 folders: train, valid, test
    """
    img_dir = src_root / split / "images"
    lbl_dir = src_root / split / "labels"
    out_img_dir = dst_root / split / "images"
    out_lbl_dir = dst_root / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    image_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
    print(f"\nProcessing {split} set: {len(image_paths)} images")
    for img_path in tqdm(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipped unreadable image: {img_path}")
            continue
        median = denoise_median(image, 0)
        processed = increase_contrast(median, 0)
        out_path = out_img_dir / img_path.name
        cv2.imwrite(str(out_path), processed)
        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            label_out_path = out_lbl_dir / label_path.name
            label_out_path.write_text(label_path.read_text())

for split in ["train", "valid", "test"]:
    process_split(split)
print("\nPreprocessing complete. Preprocessed dataset saved to 'datasets/dataset_yolo'.")
