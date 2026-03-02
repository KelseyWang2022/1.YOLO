import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import math

# 初始化YOLO模型
model = YOLO('yolov8n.pt')

# 初始化DeepSORT
max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# 打开视频文件
video_path = 'high_traffic.mov'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和尺寸
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义感兴趣区域 (ROI)
roi = [(0, height//2), (width, height//2), (width, height), (0, height)]

# 初始化计数器和字典
vehicle_count = {'up': 0, 'down': 0}
vehicle_speeds = {}

def point_in_roi(point, roi):
    return cv2.pointPolygonTest(np.array(roi, np.int32), point, False) >= 0

def calculate_speed(track, fps, ppm):
    if len(track) < 2:
        return 0
    dx = track[-1][0] - track[-2][0]
    dy = track[-1][1] - track[-2][1]
    distance = math.sqrt(dx**2 + dy**2)
    time = 1 / fps
    speed = (distance / ppm) / time * 3.6  # Convert to km/h
    return speed

def draw_roi(frame, roi):
    cv2.polylines(frame, [np.array(roi, np.int32)], True, (0, 255, 0), 2)

def draw_count(frame, vehicle_count):
    cv2.putText(frame, f"Up: {vehicle_count['up']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Down: {vehicle_count['down']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 假设每像素代表0.1米 (这需要根据实际情况调整)#每米约等于24.51像素
# #每米的实际距离在图像上表现为约13.08像素
pixels_per_meter = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLO进行目标检测
    results = model(frame)



    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            if int(cls) in [2, 5, 7]:  # 只跟踪汽车、公交车和卡车
                bbox = [x1, y1, x2 - x1, y2 - y1]
                # 创建一个虚拟的特征向量，这里使用一个简单的随机向量
                feature = np.random.rand(128)  # 128是一个常用的特征向量维度，您可能需要根据实际情况调整
                detections.append(Detection(bbox, conf, feature))

    # 更新跟踪器
    tracker.predict()
    tracker.update(detections)

    # 处理每个跟踪对象
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        track_id = track.track_id

        # 计算车辆中心点
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # 更新轨迹
        if track_id not in vehicle_speeds:
            vehicle_speeds[track_id] = {'track': [], 'speed': 0}
        vehicle_speeds[track_id]['track'].append(center)

        # 计算速度
        speed = calculate_speed(vehicle_speeds[track_id]['track'], fps, pixels_per_meter)
        vehicle_speeds[track_id]['speed'] = speed

        # 检查是否穿过ROI
        if point_in_roi(center, roi):
            if center[1] < height * 3/4:
                vehicle_count['up'] += 1
            else:
                vehicle_count['down'] += 1

        # 在帧上绘制边界框和信息
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}, Speed: {speed:.2f} km/h",
                    (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 绘制ROI和计数
    draw_roi(frame, roi)
    draw_count(frame, vehicle_count)

    # 显示结果
    cv2.imshow('Vehicle Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()