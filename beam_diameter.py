import numpy as np
import cv2

# Mở file .npy và truy xuất các thông số
calibration_data = np.load('camera_calibration.npy', allow_pickle=True).item()
ret = calibration_data['ret']
cameraMatrix = calibration_data['cameraMatrix']
dist = calibration_data['dist']
rvecs = calibration_data['rvecs']
tvecs = calibration_data['tvecs']

img = cv2.imread("v15_d10_5\\frame_1.jpg")
h,  w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

origin = dst

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
equalized_image = cv2.equalizeHist(dst)
# Áp dụng Gaussian Blur với kernel size là (5, 5)
blurred_image = cv2.medianBlur(equalized_image, 11)

# Áp dụng phân ngưỡng tự động để nhận dạng các đối tượng trong ảnh (trong trường hợp này là hình tròn)
_, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

thresh = 255 - thresh

kernel = np.ones((5, 5), np.uint8)
main_thresh = cv2.erode(thresh, kernel, iterations=4)

# Tìm các contour trên ảnh nhị phân
contours, _ = cv2.findContours(main_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

# Tìm hình elip bao quanh contour lớn nhất
ellipse = cv2.fitEllipse(largest_contour)

(center, axes, angle) = ellipse

# Lấy độ dài các trục chính
major_axis, minor_axis = axes
print(f"Beam: {major_axis, minor_axis}")

# Vẽ hình elip lên ảnh gốc
result = origin.copy()
cv2.ellipse(result, ellipse, (0, 255, 0), 2)

x, y, w, h = cv2.boundingRect(largest_contour)
# Tạo một ảnh mới để vẽ contour
contour_image = np.zeros_like(dst)

contour_image[y: y+h, x: x+w] = thresh[y: y+h, x: x+w]

contour_image = 255 - contour_image

# Tìm các contour trên ảnh nhị phân
hole_contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sắp xếp các contour theo diện tích giảm dần
contours_sorted = sorted(hole_contours, key=cv2.contourArea, reverse=True)

# Kiểm tra xem có ít nhất 2 contour trong danh sách
if len(contours_sorted) >= 2:
    # Lấy contour thứ hai trong danh sách đã sắp xếp
    second_contour = contours_sorted[1]

    # Vẽ contour thứ hai lên ảnh gốc
    new_result = contour_image.copy()
    # cv2.drawContours(result, [second_contour], -1, (0, 255, 0), 2)

    second_ellipse = cv2.fitEllipse(second_contour)

    (second_center, second_axes, second_angle) = second_ellipse

    second_major_axis, second_minor_axis = second_axes
    print(f"Lỗ: {second_major_axis, second_minor_axis}")

    cv2.ellipse(result, second_ellipse, (0, 255, 0), 2)

lo = 30

print(f"Tỉ lệ theo trục lớn: {lo*major_axis/second_major_axis} (cm)")
print(f"Tỉ lệ theo trục bé: {lo*minor_axis/second_minor_axis} (cm)")
print(f"Trung bình: {(lo*minor_axis/second_minor_axis+ lo*major_axis/second_major_axis)/2} (cm)")

# Hiển thị ảnh gốc và ảnh có hình elip
# cv2.imshow("Original Image", origin)
cv2.imshow("Ellipse Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()