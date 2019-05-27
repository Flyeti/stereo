import numpy as np
import cv2
import time

calibration_data = np.load('C:/Users/INIPSO/Desktop/calibration.npy')
mtx = calibration_data[0]
dist = calibration_data[1]
rvecs = calibration_data[2]
tvecs = calibration_data[3]

font = cv2.FONT_HERSHEY_COMPLEX

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
grayR = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('fr1adme')
cv2.createTrackbar('Distance', 'fr1adme', 0, 250, nothing)


while (True):
    trackb = cv2.getTrackbarPos('Distance', 'fr1adme')
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, None)
    frame2 = frame.copy()
    grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayL, (5, 5), 0)

    # ret, thresh_img = cv2.threshold(blur, 91, 255, cv2.THRESH_BINARY)
    # contours = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2]
    # for c in contours:
    #     cv2.drawContours(frame2, [c], -1, (0, 255, 0), 3)


    if cv2.waitKey(1) & 0xFF == ord('a'):
        grayR = grayL
        cv2.imshow('fr2ame', frame)

    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
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

    frameD = cv2.normalize(src=frameD, dst=frameD, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    frameD = np.uint8(frameD)
    #-------------------------------------
    cv2.imshow('fr1ame', frame)
    cv2.imshow('Disparity Map', frameD)

    # find Harris corners
    grayL = np.float32(grayL)
    dst = cv2.cornerHarris(grayL, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(grayL, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    for i in range(1, len(corners)):

        print(corners[i, 0])
        cv2.circle(frame2, (int(corners[i, 0]), int(corners[i, 1])), 7, (0, 255, 0), 2)

    for i in range(0, 640, 20):
        for j in range(0, 480, 5):
            if (frame2[j, i] == np.array([0, 255, 0], dtype=np.uint8)).all():
                frame[j, i] = frame2[j, i]
                if (trackb>=frameD[j,i]):
                    cv2.putText(frame, str(frameD[j, i]), (i, j), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                #cv2.putText(frame, str(5/int(frameD[j, i])*1000), (i, j), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame2', frame2)
    #cv2.imshow('frameC', frame2)
    cv2.imshow('fr1adme', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()