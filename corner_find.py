import cv2

def get_mouse_click(event, x, y, flags, param): #prints out clicked point on the image to later manually enter to the dewarping function 
    if event == cv2.EVENT_LBUTTONDOWN: #on left mouse button click
        print(f"point: ({x}, {y})")

img = cv2.imread("driving_images\im005-snow.jpg") #read the image (selected an image where the corners are easily distinguishable)
cv2.namedWindow("click corners", cv2.WINDOW_NORMAL) #create window
cv2.setWindowProperty("click corners", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #make it fullscreen
cv2.imshow("click corners", img)
cv2.setMouseCallback("click corners", get_mouse_click)
cv2.waitKey(0)