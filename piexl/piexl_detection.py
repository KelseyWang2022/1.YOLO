import cv2
import numpy as np


def calculate_pixel_distance(video_path, frame_number=0):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 设置要读取的帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # 读取帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取视频帧")
        return

    # 显示帧并等待用户选择两个点
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Frame", frame)

    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", mouse_callback)

    while len(points) < 2:
        cv2.waitKey(1)

    # 计算两点之间的距离
    distance = np.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)

    print(f"两点之间的像素距离: {distance:.2f}")

    cv2.destroyAllWindows()
    cap.release()


# 使用示例
video_path = r"C:\Users\Kelsey\PycharmProjects\computer_vision\piexl\traffic_video2.mov"
calculate_pixel_distance(video_path)


#计算方式
#每米约等于24.51像素
#每米的实际距离在图像上表现为约24.5像素