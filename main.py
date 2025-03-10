import cv2
import argparse
import os

def process(img_path):
    img = cv2.imread(img_path)
    return img

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
