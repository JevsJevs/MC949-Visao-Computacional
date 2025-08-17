"""Utility functions to be used on Jupyter notebook enviroments
"""

import cv2
import matplotlib.pyplot as plt

def show_image(img: cv2.Mat, show_axis: bool = False):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.imshow(img_rgb)
    if not show_axis:
        plt.axis('off')  # Hide axes
    plt.show()