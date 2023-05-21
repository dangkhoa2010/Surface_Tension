import numpy as np
import cv2 as cv
import glob

# chessboardSize = (11, 9)
# frameSize = (1440,1080)
#
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
# objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
#
# size_of_chessboard_squares_mm = 8
# objp = objp * size_of_chessboard_squares_mm
#
#
# # Arrays to store object points and image points from all the images.
# objpoints = [] # 3d point in real world space
# imgpoints = [] # 2d points in image plane.
#
# images = glob.glob('*.jpg')
#
# for image in images:
#
#     img = cv.imread(image)
#     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     equalized = cv.equalizeHist(gray)
#     gray = cv.GaussianBlur(equalized, (7, 7), 10)
#
#     # Find the chess board corners
#     ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
#
#     if not ret:
#         print(f"Faild {image}")
#
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners)
#
#         # # Draw and display the corners
#         # cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
#         # cv.imshow('img', img)
#         # cv.waitKey(0)
#
#
# cv.destroyAllWindows()
#
# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
# print('Calibed')
# # Tạo dict để lưu trữ các thông số calibrateCamera
# calibration_data = {
#     'ret': ret,
#     'cameraMatrix': cameraMatrix,
#     'dist': dist,
#     'rvecs': rvecs,
#     'tvecs': tvecs
# }
#
# # Lưu dict vào file .npy
# np.save('camera_calibration.npy', calibration_data)

# Mở file .npy và truy xuất các thông số
calibration_data = np.load('camera_calibration.npy', allow_pickle=True).item()
ret = calibration_data['ret']
cameraMatrix = calibration_data['cameraMatrix']
dist = calibration_data['dist']
rvecs = calibration_data['rvecs']
tvecs = calibration_data['tvecs']

img = cv.imread('8.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.jpg', dst)

mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.jpg', dst)

# mean_error = 0

# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
#
# print( "total error: {}".format(mean_error/len(objpoints)) )

