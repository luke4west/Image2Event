import cv2
import random
import os
import pandas as pd

video_list = []
for root, dirs, files in os.walk("/data/LateOrchestration/Event_Domain_Adaptation/event_data", topdown=False):
    for name in files:
        if name.endswith("dvs-video.avi"):
            print(os.path.join(root, name))
            video_list.append(os.path.join(root, name))

save_path = "/data/LateOrchestration/Event_Domain_Adaptation/event_images"
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

            # 这里可以对每一帧进行处理，例如保存为图像文件
            cv2.imwrite(f"{save_path}/video_{idx}_frame_{frame_count}.png", frame)
            event_data_list.append(f"event_images/video_{idx}_frame_{frame_count}.png")
            # print(frame.shape)

        frame_count += 1

    # 释放视频对象
    cap.release()

    print("Frames extracted successfully.")

event_df = pd.DataFrame()
event_df["event"] = event_data_list
event_df.to_csv("/data/LateOrchestration/Event_Domain_Adaptation/event_data.csv", index=False)