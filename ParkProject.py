import cv2
import torch
from datetime import datetime
import os
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from preprocessing.SingleDenoiseFunctions import denoise_median, increase_contrast
from classification_empty_spots.RuleBasedColorParkClassificationFunctions import classify_parking_slot

'''
Example of the usage of the whole system for an image
'''

empty_slot_counts = {
    "normal": 0,    # counter for empty parking lots with white lines
    "pay": 0,       # counter for empty parking lots with blue lines
    "disabled": 0,  # counter for empty parking lots with yellow lines
    "pregnant": 0,  # counter for empty parking lots with pink lines
}

input_image_path = "datasets/example_image_project.jpg"
yolo_model_path = 'models/model_yolo/weights/best.pt'
resnet_model_path = 'models/model_resnet.pt'
yolo_classes = ['space-empty', 'space-occupied']
resnet_classes = ['Illegal-parking', 'Slot-occupied']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_base = f"results/final_result_{timestamp}"
os.makedirs(output_base, exist_ok=True)

output_annotated = os.path.join(output_base, "yolo_output.jpg")
output_crops = os.path.join(output_base, "crops")
os.makedirs(output_crops, exist_ok=True)

yolo = YOLO(yolo_model_path)
resnet = torch.load(resnet_model_path, map_location=device, weights_only=False)
resnet.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# image preprocessing
image = cv2.imread(input_image_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_med = denoise_median(img, 1)
img_contrast = increase_contrast(img_med, 1)
img_contrast_bgr = cv2.cvtColor(img_contrast, cv2.COLOR_RGB2BGR)
img_rgb_clean = img_contrast
img_rgb_annotated = img_rgb_clean.copy()
img_rgb_annotated2 = img_rgb_clean.copy()

# yolo object detection and classification [occupied/not-occupied]
results = yolo.predict(img_contrast_bgr)[0]

for i, box in enumerate(results.boxes):
    cls = int(box.cls[0])
    yolo_label = yolo_classes[cls]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = (0, 255, 0) if yolo_label == 'space-empty' else (0, 0, 255)
    cv2.rectangle(img_rgb_annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_rgb_annotated, yolo_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

output_annotated = os.path.join(output_base, "yolo_annotated.jpg")
cv2.imwrite(output_annotated, cv2.cvtColor(img_rgb_annotated, cv2.COLOR_RGB2BGR))
print(f"Immagine YOLO annotata salvata in: {output_annotated}")
cv2.imshow("YOLO Annotated Image", cv2.cvtColor(img_rgb_annotated, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

for i, box in enumerate(results.boxes):
    cls = int(box.cls[0])
    yolo_label = yolo_classes[cls]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img_rgb_clean[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    crop_pil = Image.fromarray(crop)
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    # rule-based classification for empty parking slots according to the color of the parking lines
    if yolo_label == 'space-empty':
        temp_crop_path = os.path.join(output_crops, f"temp_{i:03d}.png")
        cv2.imwrite(temp_crop_path, crop_bgr)
        try:
            rule_label = classify_parking_slot(temp_crop_path)
        except Exception as e:
            print(f"Errore classificando {temp_crop_path}: {e}")
        if rule_label not in empty_slot_counts:
            empty_slot_counts[rule_label] = 0
        empty_slot_counts[rule_label] += 1
        os.remove(temp_crop_path)
        label_folder = os.path.join(output_crops, "space-empty", rule_label)
        os.makedirs(label_folder, exist_ok=True)
        filename = f"{i:03d}_{datetime.now().strftime('%H%M%S')}.png"
        crop_pil.save(os.path.join(label_folder, filename))
        continue

    # ResNet classification for occupied parking slots [legal/illegal]
    input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet(input_tensor)
        pred_class = int(torch.argmax(output, dim=1))
        pred_label = resnet_classes[pred_class]

    # set bbox color and label based on prediction [only illegal but customizable]
    combined_text = f"{yolo_label} / {pred_label}"

    if pred_label == "Illegal-parking":
        color = (255, 0, 0)  # red if wrong
        cv2.rectangle(img_rgb_annotated2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_rgb_annotated2, combined_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    """
    else:
        color = (0, 0, 255)  # blue if correct
        cv2.rectangle(img_rgb_annotated2, (x1, y1), (x2, y2), color, 0)
        cv2.putText(img_rgb_annotated2, "hello", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    """

    label_folder = os.path.join(output_crops, "space-occupied", pred_label.replace(' ', '_'))
    os.makedirs(label_folder, exist_ok=True)
    filename = f"{i:03d}_{datetime.now().strftime('%H%M%S')}.png"
    crop_pil.save(os.path.join(label_folder, filename))


cv2.imshow("Illegal parkings", cv2.cvtColor(img_rgb_annotated2, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

output_annotated2 = os.path.join(output_base, "annotated_illegal_only.jpg")
cv2.imwrite(output_annotated2, cv2.cvtColor(img_rgb_annotated2, cv2.COLOR_RGB2BGR))

print("\nEmpty parking slots summary:")
for slot_type in ["normal", "pay", "disabled", "pregnant"]:
    print(f"{slot_type}: {empty_slot_counts.get(slot_type, 0)}")
print(f"\nResults saved in: {output_base}")
