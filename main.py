import cv2
import argparse
import os
import numpy as np


# dewarps the image fixing perspective
def dewarp_image(img):
    # coordinates of image corners found by mouse clicks (code: corner_find.py)
    orig_corners = np.float32([
        (9, 15),
        (236, 5),
        (30, 244),
        (251, 236)
    ])
    # desired corner coordinates (4 corners of a 256x256 image)
    correct_corners = np.float32([
        (0, 0), (256, 0), (0, 256), (256, 256)
    ])
    # perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(orig_corners, correct_corners)
    # applying matrix
    dewarped = cv2.warpPerspective(img, matrix, (256, 256))
    return dewarped


# fills the missing region on the image
def inpaint_missing(img):
    # crops the image to only top right fourth where missing region always is
    upper_right = img[0:256//2, 256//2:256]
    grayscale = cv2.cvtColor(upper_right, cv2.COLOR_BGR2GRAY)
    # binary threshholding on a grayscaled image to prepare for edge detection
    ret, thresh = cv2.threshold(grayscale, 7, 255, cv2.THRESH_BINARY)
    # canny edge detection
    edge = cv2.Canny(thresh, 100, 150)
    # find the contours on the Canny applied image
    contours, hierarchy = cv2.findContours(
        edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # select max contour by area
    max_contour = max(contours, key=cv2.contourArea)
    # create a circle of radius and coordinates of the missing region
    (x, y), radius = cv2.minEnclosingCircle(max_contour)
    # create an all black mask of the same size as the image
    mask = np.zeros(img.shape[:2], np.uint8)
    # draw a the white circle of the missing region properties on the mask
    cv2.circle(mask, (int(x + 256//2), int(y)), int(radius + 2), (255), -1)
    # apply mask to the image and inpaint the circle
    inpainted = cv2.inpaint(img, mask, 20, cv2.INPAINT_TELEA)
    return inpainted


# denoises the image
def remove_noise(img):
    # bilateral filter to remove whatever noise we can before median blur
    denoised = cv2.bilateralFilter(img, 15, 75, 75)
    b, g, r = cv2.split(denoised)
    # median blur to the blue and green channels as they introduce most noise
    b_blurred = cv2.medianBlur(b, 3)
    g_blurred = cv2.medianBlur(g, 3)
    denoised = cv2.merge([b_blurred, g_blurred, r])
    # median blur again on the entire image to further smoothen the noise
    denoised = cv2.medianBlur(denoised, 3)
    # Non-local means coloured to remove colour spots
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 1, 9, 7, 20)
    return denoised


# balances colours according to gray world assumtion
def balance_colours(img):
    b, g, r = cv2.split(img)
    # calculate mean of all pixel brightnesses across 3 channels
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    mean_all = (b_mean + g_mean + r_mean) / 3
    # scale each channel to have equal brightness
    b = np.clip(b * (mean_all / b_mean), 0, 255).astype(np.uint8)
    g = np.clip(g * (mean_all / g_mean), 0, 255).astype(np.uint8)
    r = np.clip(r * (mean_all / r_mean), 0, 255).astype(np.uint8)
    balanced = cv2.merge([b, g, r])
    return balanced


# adjusts contrast and brightness of the image
def contrast_brightness(img):
    # very slightly lowers the contrast and brightens the image
    adjusted = cv2.convertScaleAbs(img, None, 0.97, 1)
    return adjusted


# unsharp mask sharpening method
def sharpen_image(img):
    # Gaussian blur to soften the image creating the mask
    blur_mask = cv2.GaussianBlur(img, (0, 0), 1.5)
    # sharpens the image by subtracting blurred mask from the original
    sharpened = cv2.addWeighted(img, 1 + 1.5, blur_mask, -1.5, 0)
    return sharpened


# processing pipeline, applies every function to the image
def process(img_path):
    # read the image
    img = cv2.imread(img_path)
    dewarped = dewarp_image(img)
    inpainted = inpaint_missing(dewarped)
    denoised = remove_noise(inpainted)
    balanced = balance_colours(denoised)
    contrasted = contrast_brightness(balanced)
    sharpened = sharpen_image(contrasted)
    return sharpened


# creates the results directory and writes the processsed images
def main(img_dir):
    # create the results directory if doesn not exist
    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)
    # loop through images in the given directory
    for img in os.listdir(img_dir):
        # make sure to skip any non image files such as .DS_Store
        if img.endswith((".jpg", ".jpeg")):
            # get image path
            img_path = os.path.join(img_dir, img)
            processed_img = process(img_path)
            # write processed image into results directory
            cv2.imwrite(os.path.join(results_dir, img), processed_img)


# argumet parser to let user specify the images' directory
parser = argparse.ArgumentParser()
# create image directory argument
img_dir = parser.add_argument("img_dir")
if __name__ == "__main__":
    # parse argumets
    args = parser.parse_args()
    img_dir = args.img_dir
    # call main function on the input directory
    main(img_dir)
