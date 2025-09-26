import cv2
import os

video_path = "Q1video.mp4"
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps_extract = 5
frame_count = 0
saved_count = 0

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(video_fps / fps_extract)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    
    frame_count += 1

cap.release()
print(f"共存檔 {saved_count} 張圖片到 {output_folder}")
