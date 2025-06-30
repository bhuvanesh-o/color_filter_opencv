import cv2
print(cv2.__version__)
import numpy as np


# Video cam
cap = cv2.VideoCapture(0)
cap.set(3,360) # width
cap.set(4,480) #height
cap.set(10,100) #brightness

#this is just for stacking images

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

'''
def empty(a):
    pass

# this is for trackbars

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
'''
# HSV ranges for some basic colors
color_ranges = {
    'red': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # Two ranges for red
    'green': [(np.array([36, 100, 100]), np.array([86, 255, 255]))],
    'blue': [(np.array([94, 80, 2]), np.array([126, 255, 255]))],
    'yellow': [(np.array([15, 150, 150]), np.array([35, 255, 255]))],
}

# Define BGR colors for drawing rectangles & text for each color
color_bgr = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255)
}

color_name = input("Enter color to detect (red / green / blue / yellow): ").lower()

if color_name not in color_ranges:
    print("Color not supported. Please choose from red, green, blue, yellow.")
    exit()

while True:

    success,img = cap.read()
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    '''
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    # Define the HSV range for the color you want to detect (example: red)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    '''

    # Create mask for the selected color
    masks = []
    for lower, upper in color_ranges[color_name]:
        masks.append(cv2.inRange(hsv, lower, upper))
    mask = sum(masks)  # Sum masks if multiple ranges (like for red)

    # Find contours for the color
    contours,hierarchy= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small areas (adjust this value if needed)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr[color_name], 2)
            cv2.putText(img, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)
            
    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Original Video",img)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    imgStack = stackImages(0.6,([img,hsv],[mask,result]))

    cv2.imshow("Stacked Images",imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
