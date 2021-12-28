# calibrate-camera
calibrate camera to remove lens distortion



File: calibrate.py
------------------------
Description: 
Associate single RGB image from a single camera to a single pointcloud frame for debugging purposes 

How to run:
- d / --dataset : path to calibration dataset (create dataset by taking pictures of calibration board using camera)
- i / --image    : image to remove distortion
- t / --type     : Type of board, 0 = chessboard | 1 = aruco | 2 = charuco 
```
$ python calibrate.py -d path/dataset -i imageToUndistort.jpg -t 0
```



File: create_arUco_charUco_board.py
------------------------
Description: 
Generate aruco or charuco board

How to run:
- t / --type  : 0 = aruco board | 1 = charuco board
- o/ --output : output path to board
- c / --cols  : specify number columns of board | default = 5
- r / --rows  : specify number rows of board | default = 7
```
$ python calibrate.py -t 0 -o result -c 10 -r 10
```


File: create_chessboard.py
------------------------
Description: 
Generate aruco or charuco board

How to run:
- o / --output : output file (default out.svg)
- r / --rows : pattern rows (default 11)
- c / --columns : pattern columns (default 8)
- T / --type : type of pattern, circles, acircles, checkerboard, radon_checkerboard (default circles)
- s / --square_size : size of squares in pattern (default 20.0)
- R / --radius_rate : circles_radius = square_size/radius_rate (default 5.0)
- u / --units : mm, inches, px, m (default mm)
- w / --page_width : page width in units (default 216)
- h / --page_height : page height in units (default 279)
- a / --page_size : page size (default A4), supersedes -h -w arguments
- m / --markers : list of cells with markers for the radon checkerboard
- H / --help : show help



To Do:
------------------------
  - [ ] combine aruco, charuco and chessboard all in one file
  - [ ] Convert code to C++
  - [ ] remove need for svgfig dependency in create_chessboard.py
