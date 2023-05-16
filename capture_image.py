import cv2


def Capture_image():
    # Đọc camera
    cap = cv2.VideoCapture(1)

    # Kiểm tra xem camera có khả dụng không
    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    count = 1  # Biến đếm số ảnh đã chụp

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        if not ret:
            print("Không thể nhận dạng frame từ camera.")
            break

        # Hiển thị frame trong cửa sổ
        cv2.imshow('Camera', frame)

        # Chờ bấm phím
        key = cv2.waitKey(1)

        # Nếu bấm phím space, chụp ảnh
        if key == ord(' '):
            # Tạo tên tệp tin ảnh
            image_name = f"{count}.jpg"

            # Lưu ảnh vào tệp tin
            cv2.imwrite(image_name, frame)
            print(f"Đã chụp ảnh {image_name}.")

            # Tăng biến đếm
            count += 1

        # Nếu bấm phím 'q', thoát khỏi vòng lặp
        if key == ord('q'):
            break

    # Giải phóng camera và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để chạy chương trình
Capture_image()