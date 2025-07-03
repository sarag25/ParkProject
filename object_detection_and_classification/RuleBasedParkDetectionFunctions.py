import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from preprocessing.SingleDenoiseFunctions import denoise_median, increase_contrast

'''
Functions used in RuleBasedParkDetection
'''


EMPTY_PATCH_MEAN = 130  # gray image represents empty parking lot


def is_occupied_v2(img):
    """
    Rule-based function to determine if a parking slot is occupied or not
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = denoise_median(img, 0)
    contrast = increase_contrast(blur, 0)
    edges = cv2.Canny(contrast, 30, 100)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 3.5:
                contour_score = 1
                break
    mean_brightness = np.mean(gray)
    texture_score = np.std(gray) / 255.0
    diff_score = np.abs(mean_brightness - EMPTY_PATCH_MEAN) / 255.0
    score = (
        (edge_density * 3.5) +
        (contour_score * 2.5) +
        (texture_score * 2.0) +
        (diff_score * 1.0)
    )
    return score


def find_best_threshold(confidences, truth):
    """
    Iteratively try different thresholds to find the best one to separate between space empty and space occupied
    """
    best_threshold = 0
    best_f1score = 0
    for thresh in np.arange(0.5, 4.0, 0.05):
        prediction = [1 if c > thresh else 0 for c in confidences]
        score = f1_score(truth, prediction, zero_division=0)
        if score > best_f1score:
            best_f1score = score
            best_threshold = thresh
    return best_threshold, best_f1score


def show_examples(images, y_true, y_pred, title, max_samples=5):
    """
    Shows some correct and incorrect predictions
    """
    shown = 0
    plt.figure(figsize=(15, 3))
    for idx, (img, true, pred) in enumerate(zip(images, y_true, y_pred)):
        if shown >= max_samples:
            break
        if title == "Correct" and true == pred:
            pass
        elif title == "Incorrect" and true != pred:
            pass
        else:
            continue
        plt.subplot(1, max_samples, shown + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {true}, Predicted: {pred}")
        plt.axis('off')
        shown += 1
    plt.suptitle(f"Predictions {title}")
    plt.tight_layout()
    plt.show()