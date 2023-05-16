import cv2
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re
# from skimage.measure import compare_ssim

# Đường dẫn tới video đầu vào
video_path = "output60.mp4"

# # Đường dẫn tới thư mục lưu các khung hình
# output_frames_path = "frame/"


def save_video(video_path, t=10, fps=30, volume=100, v=25, times=1, write=False):
    cap = cv2.VideoCapture(1)

    # Thiết lập kích thước khung hình và tốc độ khung hình
    width = 1280
    height = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Thiết lập codec và tên file video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = f"output{fps}_nacl_{t}_{volume}ml_v{v}_{times}.mp4"
    path = os.path.join(video_path, video_name)
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # Lấy mẫu các khung hình và ghi lại chúng vào video
    start_time = time.time()  # Bắt đầu bộ đếm thời gian
    frame_count = 0  # Khởi tạo biến đếm số khung hình đã lưu
    while True:
        ret, frame = cap.read()

        if write:
            if ret:
                out.write(frame)
                frame_count += 1  # Tăng biến đếm số khung hình
                cv2.imshow(video_path, frame)

            # Nhấn phím 'q' để thoát khỏi vòng lặp
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Kiểm tra nếu đã ghi đủ số khung hình cần thiết thì dừng vòng lặp
            if frame_count == fps * t:  # Ghi đủ time giây video
                break

        else:
            cv2.imshow(video_path, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    end_time = time.time()  # Kết thúc bộ đếm thời gian

    # Tính thời gian ghi video
    total_time = end_time - start_time
    print(f"Thời gian ghi video: {total_time:.2f} giây")

    # Giải phóng bộ nhớ và đóng video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return fps, volume, v, total_time, times


def export_frames(video_path, output_frames_path):
    # Tạo thư mục lưu các khung hình
    if not os.path.exists(output_frames_path):
        os.makedirs(output_frames_path)

    # Đọc video đầu vào
    cap = cv2.VideoCapture(video_path)

    # Lặp lại các khung hình trong video
    frame_count = 0
    while cap.isOpened():
        # Đọc khung hình từ Video
        ret, frame = cap.read()

        if not ret:
            break  # Thoát khỏi vòng lặp nếu đã đọc hết video

        if ret:
            # Lưu khung hình vào thư mục
            frame_path = os.path.join(output_frames_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1


def compute_mse(img_path, img_dir):
    # Load the source image
    src_img = cv2.imread(img_path)

    # Display the source image and allow the user to select a ROI
    roi = cv2.selectROI(src_img)
    cv2.destroyAllWindows()

    # Extract the selected ROI from the source image
    x, y, w, h = roi
    src_roi = src_img[y:y + h, x:x + w]

    # Initialize a dictionary to store the MSE values for each image
    mse_dict = {}

    # Get a sorted list of all files in the directory
    files = sorted(os.listdir(img_dir), key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Loop through all images in the directory and compare their ROIs with the source ROI
    for filename in files:
        # Load the current image
        current_img = cv2.imread(os.path.join(img_dir, filename))

        # Extract the ROI from the current image
        current_roi = current_img[y:y + h, x:x + w]

        # Calculate the MSE between the source ROI and the current ROI
        mse = np.mean(np.square(src_roi - current_roi))

        # Add the MSE value to the dictionary
        mse_dict[filename] = mse

    # Initialize a new dictionary to store the new MSE values
    new_mse_dict = mse_dict.copy()

    # Set the number of iterations to perform
    num_iterations = 7

    # Iterate over the mse_dict and calculate the new MSE values n times
    for j in range(num_iterations):
        for i, (filename, mse) in enumerate(new_mse_dict.items()):
            if i == 0 or i == len(new_mse_dict) - 1:
                # Skip the first and last element
                continue
            else:
                # Calculate the new MSE value as a weighted average of the current MSE
                # and the MSE values of the two neighboring elements
                new_mse = 0.45 * new_mse_dict[list(new_mse_dict.keys())[i - 1]] + 0.1 * mse + 0.45 * new_mse_dict[
                    list(new_mse_dict.keys())[i + 1]]

                # Update the value of mse in the new_mse_dict with the new MSE value
                new_mse_dict[filename] = new_mse

    # Create a list of integers from 0 to len(mse_values)-1 with a step of 5
    xticks = range(0, len(new_mse_dict), 100)

    # Create a list of labels for the x-axis
    xtick_labels = [str(x) for x in range(0, len(new_mse_dict), 100)]
    # Plot the MSE values as a bar chart
    plt.figure()
    plt.bar(range(len(new_mse_dict)), list(new_mse_dict.values()), align='center')
    plt.xticks(xticks, xtick_labels)
    plt.ylabel('Mean Squared Error')
    plt.title('Comparison of ROIs')
    plt.show()

    fig, axs = plt.subplots(3, 5, figsize=(20, 15))

    # Get the filenames of the 15 images with the largest SSIM values
    top_15_filenames = sorted(mse_dict, key=mse_dict.get, reverse=False)[:15]

    # Sort the filenames in ascending order
    top_15_filenames = sorted(top_15_filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Load and display the top 15 images
    for i, filename in enumerate(top_15_filenames):
        img = cv2.imread(os.path.join(img_dir, filename))
        ax = axs.flat[i]
        ax.imshow(img[:, :, ::-1])
        ax.set_title(filename)
        ax.axis('off')

    # Show the plot
    plt.show()

    # Find the indices of the local minima in the MSE values
    min_indices = []
    mse_values = list(new_mse_dict.values())
    for i in range(1, len(mse_values) - 1):
        if mse_values[i] < mse_values[i - 1] and mse_values[i] < mse_values[i + 1]:
            min_indices.append(i)

    # Create a list of the filenames corresponding to the local minima
    min_filenames = []
    for index in min_indices:
        min_filename = [k for k, v in new_mse_dict.items() if v == mse_values[index]][0]
        min_filenames.append(min_filename)

    # Sort the filenames of the local minima in ascending order
    min_filenames = sorted(min_filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Calculate the distances between consecutive local minima and compute their mean
    distances = [int(re.findall(r'\d+', min_filenames[i])[0]) - int(re.findall(r'\d+', min_filenames[i - 1])[0])
                 for i in range(1, len(min_filenames))]
    mean_distance = np.mean(distances)

    # Print the local minima and their distances
    print('Local minima:')
    for filename in min_filenames:
        print(filename)
    print('Distances:')
    for distance in distances:
        print(distance)
    print('Mean distance:', mean_distance)

    # Return the list of local minima and their distances
    return min_filenames, distances, mean_distance

#
# def compute_ssim(img_path, img_dir):
#     # Load the source image
#     src_img = cv2.imread(img_path)
#
#     # Display the source image and allow the user to select a ROI
#     roi = cv2.selectROI(src_img)
#     cv2.destroyAllWindows()
#
#     # Extract the selected ROI from the source image
#     x, y, w, h = roi
#     src_roi = src_img[y:y + h, x:x + w]
#
#     # Initialize a dictionary to store the SSIM values for each image
#     ssim_dict = {}
#
#     # Get a sorted list of all files in the directory
#     files = sorted(os.listdir(img_dir), key=lambda x: int(re.findall(r'\d+', x)[0]))
#
#     # Loop through all images in the directory and compare their ROIs with the source ROI
#     for filename in files:
#         # Load the current image
#         current_img = cv2.imread(os.path.join(img_dir, filename))
#
#         # Extract the ROI from the current image
#         current_roi = current_img[y:y + h, x:x + w]
#
#         # Calculate the SSIM between the source ROI and the current ROI
#         ssim = compare_ssim(src_roi, current_roi, multichannel=True)
#
#         # Add the SSIM value to the dictionary
#         ssim_dict[filename] = ssim
#
#     # Initialize a new dictionary to store the new MSE values
#     new_ssim_dict = ssim_dict.copy()
#
#     # Set the number of iterations to perform
#     num_iterations = 50
#
#     # Iterate over the mse_dict and calculate the new MSE values n times
#     for j in range(num_iterations):
#         for i, (filename, mse) in enumerate(new_ssim_dict.items()):
#             if i == 0 or i == len(new_ssim_dict) - 1:
#                 # Skip the first and last element
#                 continue
#             else:
#                 # Calculate the new MSE value as a weighted average of the current MSE
#                 # and the MSE values of the two neighboring elements
#                 new_mse = 0.45 * new_ssim_dict[list(new_ssim_dict.keys())[i - 1]] + 0.1 * mse + 0.45 * new_ssim_dict[
#                     list(new_ssim_dict.keys())[i + 1]]
#
#                 # Update the value of mse in the new_mse_dict with the new MSE value
#                 new_ssim_dict[filename] = new_mse
#
#     # Create a list of integers from 0 to len(ssim_values) with a step of 5
#     xticks = range(0, len(ssim_dict), 10)
#
#     # Create a list of labels for the x-axis
#     xtick_labels = [str(x) for x in range(0, len(ssim_dict), 10)]
#     # Plot the SSIM values as a bar chart
#     plt.figure()
#     plt.bar(range(len(ssim_dict)), list(ssim_dict.values()), align='center')
#     plt.xticks(xticks, xtick_labels)
#     plt.ylabel('Structural Similarity Index')
#     plt.title('Comparison of ROIs')
#     plt.show()
#
#     # fig, axs = plt.subplots(3, 5, figsize=(20, 15))
#     #
#     # # Get the filenames of the 15 images with the largest SSIM values
#     # top_15_filenames = sorted(ssim_dict, key=ssim_dict.get, reverse=True)[:15]
#     #
#     # # Sort the filenames in ascending order
#     # top_15_filenames = sorted(top_15_filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))
#     #
#     # # Load and display the top 15 images
#     # for i, filename in enumerate(top_15_filenames):
#     #     img = cv2.imread(os.path.join(img_dir, filename))
#     #     ax = axs.flat[i]
#     #     ax.imshow(img[:, :, ::-1])
#     #     ax.set_title(filename)
#     #     ax.axis('off')
#     #
#     # # Show the plot
#     # plt.show()
#
#     # Find the indices of the local maxima in the SSIM values
#     max_indices = []
#     ssim_values = list(new_ssim_dict.values())
#     for i in range(1, len(ssim_values) - 1):
#         if ssim_values[i] > ssim_values[i - 1] and ssim_values[i] > ssim_values[i + 1]:
#             max_indices.append(i)
#
#     # Create a list of the filenames corresponding to the local minima
#     max_filenames = []
#     for index in max_indices:
#         max_filename = [k for k, v in new_ssim_dict.items() if v == ssim_values[index]][0]
#         max_filenames.append(max_filename)
#
#     # Sort the filenames of the local minima in ascending order
#     max_filenames = sorted(max_filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))
#
#     # Calculate the distances between consecutive local minima and compute their mean
#     distances = [int(re.findall(r'\d+', max_filenames[i])[0]) - int(re.findall(r'\d+', max_filenames[i - 1])[0])
#                  for i in range(1, len(max_filenames))]
#     mean_distance = np.mean(distances)
#
#     # Print the local minima and their distances
#     print('Local minima:')
#     for filename in max_filenames:
#         print(filename)
#     print('Distances:')
#     for distance in distances:
#         print(distance)
#     print('Mean distance:', mean_distance)
#
#     # Return the list of local minima and their distances
#     return max_filenames, distances, mean_distance


def main():
    # video_path = "C:\\Users\\LEGION\\OneDrive\\Documents\\Study\\Square_Lab\\Surface_Tension"
    fps, volume, v, total_time, times = save_video(video_path, t=10, fps=60, volume=100, v=30, write=False, times=1)
    # # Tạo tên file mới với thời gian ghi video
    # new_file_name = f"output{fps}_nacl_{volume}ml_v{v}_{total_time:.2f}s_{times}.mp4"
    #
    # # Tạo đường dẫn đến file txt
    # name_file_path = os.path.join(video_path, "name.txt")
    #
    # # Mở file txt với mode append để ghi thêm vào cuối file
    # with open(name_file_path, "a") as f:
    #     f.write(new_file_name + "\n")

    # export_frames("output30.mp4", "output30")
    # path = "output60_h2o_10_75ml_v20_3"
    # compute_mse(path + "/frame_17.jpg", path)


if __name__ == "__main__":
    main()
