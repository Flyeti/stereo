import numpy as np
import cv2
import time

calibration_data = np.load('C:/Users/INIPSO/Desktop/calibration.npy')
mtx = calibration_data[0]
dist = calibration_data[1]
rvecs = calibration_data[2]
tvecs = calibration_data[3]

font = cv2.FONT_HERSHEY_COMPLEX

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
grayR = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#-----------------------------------
# Check if a point is inside a rectangle
def rect_contains (rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

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

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)
#-----------------------------------
while (True):
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, None)
    frame2 = frame.copy()
    grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayL, (5, 5), 0)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        grayR = grayL
        cv2.imshow('fr2ame', frame)

    # --------------SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=18 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    frameD = wls_filter.filter(displ, grayL, None, dispr)  # important to put "imgL" here!!!
    #если номер дел на сто, то выводить нумпай массив dspl
    frameD = cv2.normalize(src=frameD, dst=frameD, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    frameD = np.uint8(frameD)
    #-------------------------------------------

    #cv2.imshow('fr1ame', frame)
    cv2.imshow('Disparity Map', frameD)

    # ----------MASK-------------
    hsv = frameD.copy()
    lower = np.array([150], np.uint8)
    upper = np.array([255], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow('mask', mask)
    # ---------------------------

    #-------triangulation-----------
    # cv2.imwrite("img.jpg", grayL)
    # img = cv2.imread("image.jpg");
    # size = img.shape
    # rect = (0, 0, size[1], size[0])
    # #Create an instance of Subdiv2D with the rectangle obtained in the previous step
    # subdiv = cv2.Subdiv2D(rect)


    #----------CORNERS--------------
    corners = cv2.goodFeaturesToTrack(grayL, 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(grayL,(x, y),3, 255, -1)
        # Create an array of points.
        # subdiv.insert(i.ravel())
        # print(subdiv)

    cv2.imshow('corn', grayL)

    # Draw delaunay triangles
    # draw_delaunay(img, subdiv, (255, 255, 255))

    # cv2.imshow('corn', img)


    #--------------------------------

    # #----------COMPARE CORNERS AND MASK-------------
    # contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for contour in contours:
    #     # for i in corners:
    #     #     x, y = i.ravel()
    #     #     if (contour.any() == i.ravel()):
    #     #         cv2.circle(grayL, (x, y), 3, 0, -1)
    #     # cv2.imshow('corn', grayL)
    #     print('0', contour)
    # # for i in range(0, 640, 20):
    # #     for j in range(0, 480, 5):
    # #         if (frame2[j, i] == np.array([0, 255, 0], dtype=np.uint8)).all():
    # #             frame[j, i] = frame2[j, i]
    # #             cv2.putText(frame, str(frameD[j, i]), (i, j), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    # #             #cv2.putText(frame, str(5/int(frameD[j, i])*1000), (i, j), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    #cv2.imshow('frame2', frame2)
    #cv2.imshow('fr1adme', frame)







    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()