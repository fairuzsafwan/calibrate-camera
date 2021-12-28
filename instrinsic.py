import math

#FOV: 23 * 2 = 46 degrees -Huawei Nove lite 2

"""



K = [cx, b, ux]
    [0, cy, uy]
    [0,  0,  1]
    
cx = focal length in pixel (refer to notes No.1)
cy = focal length in pixel (refer to notes No.1)
b = represents the skew coefficient between the x and the y axis, and is often 0 
ux = represent the principal point, which would be ideally in the center of the image.
uy = represent the principal point, which would be ideally in the center of the image.

Notes: 
1. Convert Focal length from mm to pixels

Depending on how accurate you need the focal you can get a good quick estimate if you know either of the following:

CCD/CMOS sensor physical size
Horizontal field of view (FOV)
If you know the sensor's physical width in mm, then the focal in pixels is:

focal_pixel = (focal_mm / sensor_width_mm) * image_width_in_pixels

And if you know the horizontal field of view, say in degrees,

focal_pixel = (image_width_in_pixels * 0.5) / tan(FOV * 0.5 * PI/180)
"""

image_width_in_pixels = 3120
image_height_in_pixels = 4160
FOV = 46

fpx = (image_width_in_pixels * 0.5) / math.tan(FOV * 0.5 * (math.pi/180))
fpy = (image_height_in_pixels * 0.5) / math.tan(FOV * 0.5 * (math.pi/180))

print("fpx: {:.2f}".format(fpx))
print("fpy: {:.2f}".format(fpy))
k = [[fpx, 0, image_width_in_pixels/2],
     [0, fpy, image_height_in_pixels/2],
     [0, 0, 1]]

print(k)