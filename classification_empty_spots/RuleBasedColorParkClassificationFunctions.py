import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

'''
Functions used in RuleBasedColorParkClassification
'''

COLOR_RULES = [
    ("pay", (np.array([90, 20, 140]), np.array([130, 255, 255]))),  # blue lines
    ("pregnant", (np.array([150, 50, 100]), np.array([180, 255, 255]))),  # pink lines
    ("disabled", (np.array([15, 60, 170]), np.array([40, 255, 255]))),  # yellow lines
    ("normal", (np.array([0, 0, 220]), np.array([180, 40, 255])))  # white lines
]

FOLDER_TO_LABEL = {
    "pay": "pay", # blue lines
    "disabled": "disabled", # yellow lines
    "pregnant": "pregnant", # pink lines
    "normal": "normal"  # white lines
}


def classify_parking_slot(image_path, visualize=False):
    """
    Rule-based function to determine the type of empty parking slot
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_pixels = hsv.shape[0] * hsv.shape[1]

    results = []
    masks = []

    for label, (lower, upper) in COLOR_RULES:
        mask = cv2.inRange(hsv, lower, upper)
        count = np.sum(mask > 0)
        ratio = count / total_pixels
        results.append((label, ratio))
        masks.append((label, mask))
        print(f"{label}: {ratio*100:.2f}%")

    predicted_label = max(results, key=lambda x: x[1])[0]

    if visualize:
        plt.figure(figsize=(12, 8))
        for i, (label, mask) in enumerate(masks):
            masked_img = cv2.bitwise_and(image, image, mask=mask)
            masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 2, i+1)
            plt.imshow(masked_img_rgb)
            plt.title(label)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return predicted_label


def show_example_with_mask_and_hist(results):
    """
    For each class, shows a randomly selected correctly classified image
    along with its mask and corresponding color histogram.
    - Hue histogram for: pay, disabled, pregnant
    - Saturation histogram for: normal
    """
    correct_by_class = {label: [] for label in FOLDER_TO_LABEL.values()}

    for image_path, expected, predicted, is_correct in results:
        if is_correct:
            correct_by_class[expected].append((image_path, expected, predicted))

    for expected, examples in correct_by_class.items():
        if not examples:
            continue

        image_path, expected, predicted = random.choice(examples)
        image = cv2.imread(image_path)
        if image is None:
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = None
        for label, (lower, upper) in COLOR_RULES:
            if label == expected:
                mask = cv2.inRange(hsv, lower, upper)
                masked = cv2.bitwise_and(image, image, mask=mask)
                break

        if mask is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

        if expected in ['pay', 'disabled', 'pregnant']:
            channel = hsv[:, :, 0]  # hue
            hist_title = "Hue Histogram"
            bins = 180
            range_vals = [0, 180]
            hist_xlabel = "Hue"
            hist_color = "purple"
        else:
            channel = hsv[:, :, 1]  # saturation
            hist_title = "Saturation Histogram"
            bins = 256
            range_vals = [0, 256]
            hist_xlabel = "Saturation"
            hist_color = "blue"

        hist = cv2.calcHist([channel], [0], mask, [bins], range_vals)
        hist = hist.flatten()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(masked_rgb)
        axes[1].set_title("Masked Image")
        axes[1].axis('off')

        axes[2].plot(hist, color=hist_color)
        axes[2].set_title(hist_title)
        axes[2].set_xlim(range_vals)
        axes[2].set_xlabel(hist_xlabel)

        plt.suptitle(f"[CORRECT] Expected: {expected} - Predicted: {predicted}", fontsize=14)
        plt.tight_layout()
        plt.show()