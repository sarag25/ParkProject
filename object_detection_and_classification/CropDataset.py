import cv2
from pathlib import Path
from tqdm import tqdm

'''
From the output of the YOLO model creates crops of single bounding boxes
dividing between the 2 classes: space-empty & space-occupied
'''

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent
base_dir = project_dir / ...    # output of the YOLO model that needs to be cropped into single images
output_base_dir = project_dir / 'datasets' / 'cropped_slots'    # cropped images
splits = ['train', 'valid', 'test']
class_dirs = {
    0: 'space-empty',
    1: 'space-occupied'
}

def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """
    Convert YOLO annotations into a format that can be used with OpenCV
    """
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

for split in splits:
    img_dir = base_dir / split / 'images'
    label_dir = base_dir / split / 'labels'
    label_files = sorted(label_dir.glob('*.txt'))
    for label_file in tqdm(label_files, desc=f"Processing {split}"):
        image_file = img_dir / label_file.with_suffix('.jpg').name
        img = cv2.imread(str(image_file))
        img_h, img_w = img.shape[:2]
        with open(label_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, w, h = map(float, parts[1:])
            x1, y1, x2, y2 = yolo_to_bbox(x_center, y_center, w, h, img_w, img_h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            save_dir = output_base_dir / split / class_dirs[class_id]
            save_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"{image_file.stem}_obj{i}_class{class_id}.jpg"
            crop_path = save_dir / crop_name
            success = cv2.imwrite(str(crop_path), crop)
print("Cropping completed.")