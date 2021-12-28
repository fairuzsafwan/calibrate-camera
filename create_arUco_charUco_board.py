import numpy as np
import cv2
import cv2.aruco as aruco
import pathlib
import argparse
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True,
	help="0 = aruco board | 1 = charuco board")
ap.add_argument("-o", "--output", required=True,
	help="output path to board")
ap.add_argument("-c", "--cols", required=False, default = "5",
	help="[Optional] specify number columns of board")
ap.add_argument("-r", "--rows", required=False, default = "7",
	help="[Optional] specify number rows of board")
args = vars(ap.parse_args())

boardType = args["type"]
outputPath = args["output"]
cols = int(args["cols"])
rows = int(args["rows"])
type_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)

#Create folder if it doesn't exist
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

def createAruco():
    marker_length_aruco = 3
    marker_separation_aruco = 0.25
    aruco_size = (1000,1000)
    
    arucoParams = aruco.DetectorParameters_create()
    aruco_board = aruco.GridBoard_create(cols, rows, marker_length_aruco, marker_separation_aruco, type_dict)
    img_aruco = aruco_board.draw(aruco_size)
    cv2.imwrite(os.path.join(outputPath, "aruco_board.jpg"), img_aruco)

def createCharuco():
    square_length_charuco = 3.2
    marker_length_charuco = 2
    charuco_size = (1000,1000)
    
    charuco_board = aruco.CharucoBoard_create(cols, rows, square_length_charuco, marker_length_charuco, type_dict)
    img_charuco = charuco_board.draw(charuco_size)
    cv2.imwrite(os.path.join(outputPath, "charuco_board.jpg"), img_charuco)


if boardType == "0":
    createAruco()
    print("Aruco board successfully created!")
elif boardType == "1":
    createCharuco()
    print("Charuco board successfully created!")
else:
    sys.exit("Error: invalid board type detected!")


