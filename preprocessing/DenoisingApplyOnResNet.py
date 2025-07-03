import cv2
from pathlib import Path
from tqdm import tqdm
from SingleDenoiseFunctions import increase_contrast, denoise_median

'''
Function used in preprocessing to apply denoising to the raw dataset for classification of occupied parking slots
and then save the result as dataset_resnet
'''

src_root = Path("...")  # raw dataset before denoising
dst_root = Path("../datasets/dataset_resnet")   # dataset after denoising


def process_split(split):
    """
    Apply the procedure to all 3 splits: train, valid, test;
    Each containing subfolders like Slot-occupied, Illegal-parking
    """
    split_dir = src_root / split
    for category_dir in split_dir.iterdir():
        if not category_dir.is_dir():
            continue

        out_category_dir = dst_root / split / category_dir.name
        out_category_dir.mkdir(parents=True, exist_ok=True)

        image_paths = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")) + list(category_dir.glob("*.jpeg"))
        print(f"\nProcessing {split}/{category_dir.name}: {len(image_paths)} images")

        for img_path in tqdm(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Skipped unreadable image: {img_path}")
                continue
            median = denoise_median(image, 0)
            processed = increase_contrast(median, 0)
            out_path = out_category_dir / img_path.name
            cv2.imwrite(str(out_path), processed)

for split in ["train", "valid", "test"]:
    process_split(split)

print("\nPreprocessing complete. Preprocessed dataset saved to 'datasets/dataset_resnet'.")