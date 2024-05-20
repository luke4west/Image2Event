import cv2
import random
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance

def is_frame_mostly_blank(frame, threshold=192):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    variance = np.var(gray_frame)
    return variance < threshold

def crop_image(image, y_start=23, y_end=242):
    cropped_image = image[y_start:y_end, :]
    return cropped_image

def sharpen_and_enhance(image, sharpness_factor=5.0, contrast_factor=1):
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(sharpness_factor)
    
    enhancer = ImageEnhance.Contrast(sharpened_image)
    enhanced_image = enhancer.enhance(contrast_factor)
    
    return enhanced_image

def process_image(frame, output_path,event_data_list,idx,frame_count):
    try:
        # Load the image
        image = frame

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to smooth the image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply Non-Local Means Denoising with adjusted parameters
        h = 5 # Filter strength, reduce this value to preserve more details
        template_window_size = 7  # Size of the window used to compute weighted average for a given pixel
        search_window_size = 14  # Size of the window used to search for similar blocks

        denoised = cv2.fastNlMeansDenoising(blurred, None, h, template_window_size, search_window_size)

        # Convert the denoised image to PIL format for enhancement
        denoised_pil = Image.fromarray(denoised)

        # Sharpen and enhance the image
        enhanced_image = sharpen_and_enhance(denoised_pil)

        # Convert the enhanced image back to OpenCV format
        enhanced_cv = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

        if not is_frame_mostly_blank(enhanced_cv):
            # Save the enhanced image to the output directory with the same filename
            cv2.imwrite(output_path, crop_image(enhanced_cv))
            event_data_list.append(f"event_images/video_{idx}_frame_{frame_count}.png")
        
        return event_data_list
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return False

video_list = []
for root, dirs, files in os.walk("/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_data", topdown=False):
    for name in files:
        if name.endswith("dvs-video.avi"):
            print(os.path.join(root, name))
            video_list.append(os.path.join(root, name))

save_path = "/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_images"
os.makedirs(save_path, exist_ok=True)

# Set random seed
random.seed(2024)
    
event_data_list = []
for idx, video_file in enumerate(video_list):        
    # Read video file
    cap = cv2.VideoCapture(video_file)

    # Check if video successfully opened
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # Get video frame rate and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate number of frames to extract (20%)
    num_frames_to_extract = int(total_frames * 0.2)

    sample_frames = list(range(total_frames))
    random.shuffle(sample_frames)
    sample_frames = sample_frames[0:num_frames_to_extract]

    # Set counter
    frame_count = 0

    # Read video until extracted frames are reached
    while frame_count < total_frames:
        
        if frame_count in sample_frames:
            ret, frame = cap.read()

            # Check if frame successfully read
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Check if the frame is mostly blank
            # if not is_frame_mostly_blank(frame, threshold=50):
            image_path = f"{save_path}/video_{idx}_frame_{frame_count}.png"
            event_data_list_appended = process_image(frame, image_path,event_data_list,idx,frame_count)


        frame_count += 1

    # Release video object
    cap.release()

    print("Frames extracted successfully.")

event_df = pd.DataFrame()
event_df["event"] = event_data_list_appended
event_df.to_csv("/home/ruiyang/Projects/v2e/Image2Event/data/LateOrchestration/Event_Domain_Adaptation/event_data.csv", index=False)
