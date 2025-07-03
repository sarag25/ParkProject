import os
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from RuleBasedParkDetectionFunctions import is_occupied_v2, find_best_threshold, show_examples

'''
Apply rule based parking detection on the dataset CNRPark
'''

metadata = pd.read_csv('../datasets/dataset_cnrpark/CNRPark+EXT.csv', low_memory=False)
metadata['image_path'] = metadata['image_url'].str.replace('CNR-EXT', 'CNR-EXT-Patches-150x150', regex=False)
patch_rows = metadata[metadata['image_path'].str.contains('PATCHES')]
patch_rows = patch_rows.dropna(subset=['occupancy']).reset_index(drop=True)
confidences = []
y_true = []
images = []

for _, row in tqdm(patch_rows.iterrows(), total=len(patch_rows)):
    full_path = os.path.join("../datasets/dataset_cnrpark", row['image_path'])
    image = cv2.imread(full_path)
    if image is None:
        continue
    confidence = is_occupied_v2(image)
    confidences.append(confidence)
    y_true.append(int(row['occupancy']))
    images.append(image)

best_thresh, best_f1 = find_best_threshold(confidences, y_true)
print(f"Best threshold: {best_thresh:.2f} (F1 = {best_f1:.3f})")

y_pred = [1 if c > best_thresh else 0 for c in confidences]
print("\nFinal result:")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
print(f"F1 Score:  {f1_score(y_true, y_pred, zero_division=0):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

show_examples(images, y_true, y_pred, title="Correct")
show_examples(images, y_true, y_pred, title="Incorrect")
