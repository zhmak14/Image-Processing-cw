import cv2


# prints out clicked point on the image
def get_mouse_click(event, x, y, flags, param):
    # on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"point: ({x}, {y})")


# read the image (selected one where the corners are easily visible)
img = cv2.imread("driving_images/im005-snow.jpg")
# create window
cv2.namedWindow("click corners", cv2.WINDOW_NORMAL)
# make it fullscreen
cv2.setWindowProperty(
    "click corners", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
)
cv2.imshow("click corners", img)
cv2.setMouseCallback("click corners", get_mouse_click)
cv2.waitKey(0)
