import numpy as np
import cv2
import time
import math

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

calibration_data = np.load('C:/Users/INIPSO/Desktop/calibration.npy')
mtx = calibration_data[0]
dist = calibration_data[1]
rvecs = calibration_data[2]
tvecs = calibration_data[3]

flag = False
font = cv2.FONT_HERSHEY_COMPLEX
mouseX, mouseY = 10, 10

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.undistort(frame, mtx, dist, None, None)
# grayR = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
grayR = cv2.imread('right.jpg')
cv2.imshow('right', grayR)
# cv2.imwrite('right.jpg', grayR)

def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frameD,(x, y), 2, (255, 0, 0), -1)
        mouseX, mouseY = x, y

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')

#-----------------------------------
while (True):
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, None)
    frame2 = frame.copy()
    # grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayL = cv2.imread('left.jpg')
    # blur = cv2.GaussianBlur(grayL, (5, 5), 0)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        grayR = grayL
        cv2.imshow('right', grayR)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        # grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayL = cv2.imread('left.jpg')
        # cv2.imshow('left', grayL)
        # cv2.imwrite('left.jpg', grayL)
        break

while (True):

    # --------------SGBM-----------------
    window_size = 5
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,
        blockSize=7,
        P1=8 * 3 * window_size ** 2,
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
    frameD = cv2.normalize(src=frameD, dst=frameD, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    frameD = np.uint8(frameD)

    cv2.imshow('Disparity Map', frameD)
    cv2.imwrite('D.jpg', frameD)
    frame2 = frameD
    disp = frameD

    h, w = grayL.shape[:2]
    f = 0.8*w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h],  # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f],  # so that y-axis looks up
                    [0, 0, 1,      0]])

    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(grayL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()

    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()