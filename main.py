import cv2
import argparse
import os
import numpy as np

def dewarp_image(img): #dewarps the image and fixes perspective
    orig_corners = np.float32([ #coordinates of image corners found by mouse clicks (code: corner_find.py)
        (9, 15), 
        (236, 5),
        (30, 244),
        (251, 236)
    ])
    correct_corners = np.float32([ #desired corner coordinates (4 corners of a 256x256 image)
        (0, 0), (256, 0), (0, 256), (256, 256)
    ])
    matrix = cv2.getPerspectiveTransform(orig_corners, correct_corners) #perspective transformation matrix
    dewarped = cv2.warpPerspective(img, matrix, (256, 256)) #applying matrix
    return dewarped

def inpaint_missing(img):
    upper_right = img[0:256//2, 256//2:256]
    grayscale = cv2.cvtColor(upper_right, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayscale, 7, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(thresh, 100, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (int(x + 256//2), int(y)), int(radius + 2), (255), -1)
    inpainted = cv2.inpaint(img, mask, 20, cv2.INPAINT_TELEA)
    return inpainted

def process(img_path): #applies all the processing function to the image
    img = cv2.imread(img_path) #read the image
    dewarped = dewarp_image(img)
    inpainted = inpaint_missing(dewarped)
    return inpainted

def main(img_dir):
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True) #create the results directory if doesn not exist
    for img in os.listdir(img_dir): #loop through images in the given directory
        if img.endswith((".jpg", ".jpeg")): #make sure to skip any non image files such as .DS_Store
            img_path = os.path.join(img_dir, img) #get image path
            processed_img = process(img_path) 
            cv2.imwrite(os.path.join(results_dir, img), processed_img) #write processed image into results directory

parser = argparse.ArgumentParser()
img_dir = parser.add_argument("img_dir") #create image directory argument
if __name__ == "__main__":
    args = parser.parse_args() #parse argumets
    img_dir = args.img_dir 
    main(img_dir)
