import cv2
import argparse
import os
import numpy as np

def dewarp_image_fixed(img):
    orig_corners = np.float32([
        (9, 15),
        (236, 5),
        (30, 244),
        (251, 236)
    ])
    correct_corners = np.float32([
        (0, 0), (256, 0), (0, 256), (256, 256)
    ])
    matrix = cv2.getPerspectiveTransform(orig_corners, correct_corners)
    dewarped = cv2.warpPerspective(img, matrix, (256, 256))
    return dewarped

def process(img_path):
    img = cv2.imread(img_path)
    dewarped = dewarp_image_fixed(img)  # Apply dewarping
    return dewarped

def main(img_dir):
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    for img in os.listdir(img_dir):
        if img.endswith((".jpg", ".jpeg")):
            img_path = os.path.join(img_dir, img)
            processed_img = process(img_path)
            cv2.imwrite(os.path.join(results_dir, img), processed_img)

parser = argparse.ArgumentParser()
img_dir = parser.add_argument("img_dir")
if __name__ == "__main__":
    args = parser.parse_args()
    img_dir = args.img_dir
    main(img_dir)
