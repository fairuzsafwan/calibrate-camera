import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math
import argparse
#%matplotlib nbagg

def read_chessboards(images):
    """
    Charuco base pose estimation.
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1

    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")
    
    image_width_in_pixels = imsize[1]#3120
    image_height_in_pixels = imsize[0]#4160
    print("width: ", image_width_in_pixels)
    print("height: ", image_height_in_pixels)
    #print()
    FOV = 46

    fpx = (image_width_in_pixels * 0.5) / math.tan(FOV * 0.5 * (math.pi/180))
    fpy = (image_height_in_pixels * 0.5) / math.tan(FOV * 0.5 * (math.pi/180))

    print("fpx: {:.2f}".format(fpx))
    print("fpy: {:.2f}".format(fpy))
    k = np.array([[fpx, 0., image_width_in_pixels/2.],
     [0., fpy, image_height_in_pixels/2.],
     [0., 0., 1.]])

    
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])
    cameraMatrixInit = k
    print(cameraMatrixInit)
    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to calibration image dataset")
ap.add_argument("-i", "--image", required=True,
	help="image path to undistort")
ap.add_argument("-t", "--type", required=True,
	help="type of board, 0 = chessboard | 1 = aruco | 2 = charuco")
args = vars(ap.parse_args())

datasetPath = args["dataset"]
imgPath = args["image"]
boardType = args["type"]

workdir = "\\"
#datadir = "Huawei_Nova lite_calibration_images\\charuco_calibration_images\\"
datadir = datasetPath #os.path.join(datasetPath, "\\") #"cam1\\charuco\\"
square_length_charuco = 3.2
marker_length_charuco = 2
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
#queryPath = "Huawei_LXDN2_images\\"
queryPath = "\\"
fileName = imgPath
basename = os.path.basename(fileName)
fileExt = basename.split(".")[1]
output_path = os.path.dirname(fileName)


"""
#Show calibration images
im = PIL.Image.open(images[0])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(im)
#ax.axis('off')
plt.show()
"""

board = aruco.CharucoBoard_create(5, 7, square_length_charuco, marker_length_charuco, aruco_dict)
images = np.array([datadir + f for f in os.listdir(datadir) if f.endswith(".jpg") ])
order = np.argsort([int(p.split(".")[-2].split("-")[-1]) for p in images])
images = images[order]
print(images)

if boardType == "0":
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    counter = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Image path: ", fname)
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            counter += 1
            cv2.imshow('img', img)
            cv2.waitKey()
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    img_distorted = cv2.imread(imgPath)
    h,  w = img_distorted.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    print("dist: ", dist)
    print("ret: ", ret)
    print("mtx: ", mtx)
    print("newcameramtx: ", newcameramtx)
    print("counter: ", counter)
    #print("objpoints: ", objpoints)
    print("objpoints len: ", len(objpoints))
    #print("imgpoints: ", imgpoints)
    print("imgpoints len: ", len(imgpoints))
    
    
    #Method 1 to undistort
    # undistort
    dst1 = cv2.undistort(img_distorted, mtx, dist, None, newcameramtx)
    cv2.imwrite(os.path.join(output_path, "undistorted_chessboard_1_uncropped_" + basename), dst1)
    # crop the image
    x, y, w, h = roi
    dst1 = dst1[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_path, "undistorted_chessboard_1_cropped_" + basename), dst1)
    
    #Method 2 to undistort
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst2 = cv2.remap(img_distorted, mapx, mapy, cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_path, "undistorted_chessboard_2_uncropped_" + basename), dst2)
    # crop the image
    x, y, w, h = roi
    dst2 = dst2[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_path, "undistorted_chessboard_2_cropped_" + basename), dst2)
    
    i=0 # select image id
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img_distorted)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(dst1)
    plt.title("CI Method 1")
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(dst2)
    plt.title("CI Method 2")
    plt.axis("off")
    plt.show()

elif boardType == "1" or boardType == "2":

    allCorners,allIds,imsize=read_chessboards(images)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners,allIds,imsize)

    print("ret: {0}".format(ret))
    print("mtx: {0}".format(mtx))
    print("dist: {0}".format(dist))

    img_distorted = cv2.imread(imgPath)
    # print(img_distorted.shape)
    # cv2.imshow("test", img_distorted)
    # cv2.waitKey(0)
    img_undist = cv2.undistort(img_distorted,mtx,dist,None)
    
    if boardType == "1":
        cv2.imwrite(os.path.join(output_path, "undistorted_aruco_" + basename), img_undist)
    elif boardType == "2":
        cv2.imwrite(os.path.join(output_path, "undistorted_charuco_" + basename), img_undist)
    else:
        sys.exit("Error: Invalid board type, please specify correct board type, 0=chessboard, 1=aruco or charuco!")

 
    i=0 # select image id
    plt.figure()
    #frame = cv2.imread(images[i])
    #img_undist = cv2.undistort(frame,mtx,dist,None)
    plt.subplot(1,2,1)
    plt.imshow(img_distorted)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()

else:
    sys.exit("Error: Invalid board type, please specify correct board type, 0=chessboard, 1=aruco or charuco!")
