import cv2
from matplotlib import pyplot as plt
import numpy as np


def denoise_median(img, show):
    """
    Apply median filter on image and optionally show the result
    """
    dst = cv2.medianBlur(img, 5)
    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(dst), plt.title('Denoised Image (Median Filter)')
        plt.axis('off')
        plt.show()
    return dst

def denoise_gaussian(img, show):
    """
    Apply gaussian filter on image and optionally show the result
    """
    dst = cv2.GaussianBlur(img, (5, 5), 1.5)
    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(dst), plt.title('Denoised Image (Gaussian Filter)')
        plt.axis('off')
        plt.show()
    return dst

def increase_luminosity(img, show):
    """
    Increase the luminosity of the image and optionally show the result
    """
    factor = 1.2
    dst = cv2.addWeighted(img, factor, np.zeros(img.shape, img.dtype), 0, 0)
    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(dst), plt.title('Image (Increased Luminosity)')
        plt.axis('off')
        plt.show()
    return dst

def increase_contrast(img, show):
    """
    Increase the contrast of the image with histogram equalization and optionally show the result
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    lab_clahe = cv2.merge((cl, a, b))
    dst = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(dst), plt.title('Image (Increased Contrast)')
        plt.axis('off')
        plt.show()
    return dst

def increase_saturation(img, show):
    """
    Increase the saturation of the image and optionally show the result
    """
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    image[:, :, 1] = image[:, :, 1] * 1.7
    dst = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    if show:
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(img), plt.title('Original Image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(dst), plt.title('Image (Increased Saturation)')
        plt.axis('off')
        plt.show()
    return dst
