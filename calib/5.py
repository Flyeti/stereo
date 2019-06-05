import numpy as np
import cv2
import time
import math
from random import randint


calibration_data = np.load('C:/Users/INIPSO/Desktop/calibration.npy')
mtx = calibration_data[0]
dist = calibration_data[1]
rvecs = calibration_data[2]
tvecs = calibration_data[3]

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.undistort(frame, mtx, dist, None, None)
grayR = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fr = frame

tracker = cv2.TrackerCSRT_create()
bboxes = []
colors = []
points = []

mouseX, mouseY = 10, 10

def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        # cv2.circle(grayL,(x, y), 50, (255, 0, 0), -1)
        mouseX, mouseY = x, y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(fr, (x, y), 5, 0, -1)
        bbox = (x - 10, y - 10, 20, 20)
        bboxes.append(bbox)
        colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
        points.append((int(x), int(y)))
        # print('fdfd', bboxes)
        mouseX, mouseY = 10, 10

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, 3, 0)

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, 8, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, 8, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, 8, 0)


corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
corners = np.int0(corners)

# ----------CORNERS--------------
while (True):
    for i in corners:
        x, y = i.ravel()
        cv2.circle(fr, (x, y), 3, 255, -1)
        if (math.sqrt(abs(x - mouseX)) + math.sqrt(abs(y - mouseY)) <= math.sqrt(10)):
            cv2.circle(fr, (x, y), 5, 0, -1)
            bbox = (x-10, y-10, 20, 20)
            bboxes.append(bbox)
            colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
            points.append((int(x), int(y)))
            # print('fdfd', bboxes)
            mouseX, mouseY = 10, 10
    cv2.setMouseCallback('frMTame', draw_circle)
    cv2.imshow('frMTame', fr)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break


# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(tracker, fr, bbox)
    print('b', bbox)

#----------------MAIN-------------------
while (True):
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, None)
    frame2 = frame.copy()
    grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # if cv2.waitKey(1) & 0xFF == ord('a'):
    #     grayR = grayL
    #     gray = grayR
    #     cv2.imshow('fr2ame', frame)

    # -----------------TRACKING-----------------
    success, boxes = multiTracker.update(frame2)
    print('jbkj,kj', boxes)
    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame2, p1, p2, colors[i], 2, 1)
        print(i, p1, p2)

    # show frame
    cv2.imshow('MultiTracker', frame2)


    # --------------SGBM Parameters -----------------
    # window_size = 5
    # left_matcher = cv2.StereoSGBM_create(
    #     minDisparity=0,
    #     numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    #     blockSize=7,
    #     P1=8 * 3 * window_size ** 2,
    #     P2=18 * 3 * window_size ** 2,
    #     disp12MaxDiff=1,
    #     uniquenessRatio=15,
    #     speckleWindowSize=0,
    #     speckleRange=2,
    #     preFilterCap=63,
    #     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # )
    # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # # FILTER Parameters
    # lmbda = 80000
    # sigma = 1.2
    # visual_multiplier = 1.0
    # wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)
    # displ = left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
    # dispr = right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
    # displ = np.int16(displ)
    # dispr = np.int16(dispr)
    # frameD = wls_filter.filter(displ, grayL, None, dispr)  # important to put "imgL" here!!!
    # frameD = cv2.normalize(src=frameD, dst=frameD, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    # frameD = np.uint8(frameD)
    # # --------------------------------------------
    #
    # # cv2.imshow('Disparity Map', frameD)

    # ------------TRIANGULATION--------------
    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Turn on animation while drawing triangles
    animate = False

    # Define colors for drawing.
    delaunay_color = (0, 0, 0)
    points_color = (0, 0, 0)

    # Read in the image.
    img = frame;

    # Keep a copy around
    img_orig = img.copy()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)


    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (0, 0, 0));
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    # Draw delaunay triangles
    draw_delaunay(img, subdiv, (0, 0, 0));

    # Draw points
    for p in points:
        draw_point(img, p, (0, 0, 255))

    # Show results
    cv2.imshow(win_delaunay, img)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()