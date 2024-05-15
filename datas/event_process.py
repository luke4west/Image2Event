import cv2
import random
import os
import pandas as pd
import numpy as np

def is_frame_mostly_blank(frame, threshold=50):
    """
    Check if a frame is mostly blank by calculating the variance of its pixel values.
    If the variance is below the threshold, the frame is considered mostly blank.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    variance = np.var(gray_frame)
    return variance < threshold

video_list = []
for root, dirs, files in os.walk("/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_data", topdown=False):
    for name in files:
        if name.endswith("dvs-video.avi"):
            print(os.path.join(root, name))
            video_list.append(os.path.join(root, name))

save_path = "/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_images"
os.makedirs(save_path, exist_ok=True)

# 设置随机种子
random.seed(2024)
    
event_data_list = []
for idx, video_file in enumerate(video_list):        
    # 读取视频文件
    cap = cv2.VideoCapture(video_file)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # 获取视频帧率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算需要提取的帧数（20%）
    num_frames_to_extract = int(total_frames * 0.2)

    sample_frames = list(range(total_frames))
    random.shuffle(sample_frames)
    sample_frames = sample_frames[0:num_frames_to_extract]

    # 设置计数器
    frame_count = 0

    # 读取视频直到达到提取的帧数
    while frame_count < total_frames:
        
        if frame_count in sample_frames:
            ret, frame = cap.read()

            # 检查帧是否成功读取
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Check if the frame is mostly blank
            if not is_frame_mostly_blank(frame, threshold=50):
                image_path = f"{save_path}/video_{idx}_frame_{frame_count}.png"
                cv2.imwrite(image_path, frame)
                event_data_list.append(f"event_images/video_{idx}_frame_{frame_count}.png")

        frame_count += 1

    # 释放视频对象
    cap.release()

    print("Frames extracted successfully.")

event_df = pd.DataFrame()
event_df["event"] = event_data_list
event_df.to_csv("/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_data.csv", index=False)
