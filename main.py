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

def inpaint_missing(img): #fills the missing region on the image
    upper_right = img[0:256//2, 256//2:256] #crops the image to only top right fourth, knowing that the missing region can only appear there
    grayscale = cv2.cvtColor(upper_right, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grayscale, 7, 255, cv2.THRESH_BINARY) #binary threshholding on a grayscaled image to prepare for edge detection
    edge = cv2.Canny(thresh, 100, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #find the contours on the Canny applied image
    max_contour = max(contours, key=cv2.contourArea) #select max contour by area
    (x, y), radius = cv2.minEnclosingCircle(max_contour) #create a circle of radius and with center coordinates of the missing region contour
    mask = np.zeros(img.shape[:2], np.uint8) #create an all black mask of the same size as the image
    cv2.circle(mask, (int(x + 256//2), int(y)), int(radius + 2), (255), -1) #draw a the white circle of the missing region properties on the mask 
    inpainted = cv2.inpaint(img, mask, 20, cv2.INPAINT_TELEA) #apply mask to the image and inpaint the circle
    return inpainted

def remove_noise(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_mask = grayscale > 200
    bright_mask = bright_mask.astype(np.float32)[:, :, None]
    denoised_default = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 3, 20)
    denoised_bright = cv2.fastNlMeansDenoisingColored(img, None, 25, 25, 5, 20)
    denoised = (denoised_default * (1 - bright_mask) + denoised_bright * bright_mask).astype(np.uint8)
    return denoised

def correct_white(img):
    # b, g, r = cv2.split(img)
    # b_mean = np.mean(b)
    # g_mean = np.mean(g)
    # r_mean = np.mean(r)
    # mean_all = (b_mean + g_mean + r_mean) / 3
    # b = np.clip(b * (mean_all / b_mean), 0, 255).astype(np.uint8)
    # g = np.clip(g * (mean_all / g_mean), 0, 255).astype(np.uint8)
    # r = np.clip(r * (mean_all / r_mean), 0, 255).astype(np.uint8)
    # balanced = cv2.merge([b, g, r])
    # return balanced
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    l = cv2.normalize(l, None, 0, 254, cv2.NORM_MINMAX)
    lab_img_adjusted = cv2.merge([l, a, b])
    return cv2.cvtColor(lab_img_adjusted, cv2.COLOR_LAB2BGR)

# def fix_contrast_brightness(img, threshold=220, clip_limit=10.0, tile_grid_size=(8,8)):
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     overexposed_mask = l > threshold  # Pixels above threshold are overexposed
#     overexposed_mask = overexposed_mask.astype(np.uint8) * 255  # Convert to 0-255 mask
#     clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
#     l_clahe = clahe.apply(l)
#     l_fixed = np.where(overexposed_mask == 255, l_clahe, l)
#     lab_fixed = cv2.merge([l_fixed, a, b])
#     return cv2.cvtColor(lab_fixed, cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    img_smooth = cv2.bilateralFilter(img, 7, 50, 50)
    laplacian = cv2.Laplacian(img_smooth, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(img - laplacian)
    return sharpened

def process(img_path): #applies all the processing function to the image
    img = cv2.imread(img_path) #read the image
    dewarped = dewarp_image(img)
    inpainted = inpaint_missing(dewarped)
    balanced = correct_white(inpainted)
    denoised = remove_noise(balanced)
    #sharpened = sharpen_image(denoised)
    #contrasted = fix_contrast_brightness(denoised)
    return denoised

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
