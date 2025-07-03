import cv2
from preprocessing.SingleDenoiseFunctions import denoise_median, increase_luminosity, denoise_gaussian, increase_contrast, increase_saturation

'''
Try all the denoise functions on an image to see the results
'''

img_data = "../datasets/example_image_with_noise.jpg"
image = cv2.imread(img_data)
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_median = denoise_median(img_rgb, 1)
img_gaussian = denoise_gaussian(img_rgb, 1)
img_luminosity = increase_luminosity(img_rgb, 1)
img_contrast = increase_contrast(img_rgb, 1)
img_saturated = increase_saturation(img_rgb, 1)
