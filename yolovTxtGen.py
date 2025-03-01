import os
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np

INPUT_VIDEO = "videos/1.mp4"
OUTPUT_TXT = "mp4_yolo_txt/1.txt "

# yolov text file generator
def yolovTextgen(video_name,output_txt_path):
    
    
    print("Generating text file .................")
      # Replace with your actual video name
    min_conf = 0.5  # Minimum confidence threshold for detections

    # Load the pre-trained YOLOv5 model (small version)
    model = YOLO("yolov5su.pt")

    # Open the video
    cap = cv2.VideoCapture(video_name)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Couldn't open the video.")
        exit()

    frame_num = 0

    # Open a file to write the detections
    
    with open(output_txt_path, 'w') as output_file:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print('Failed to read frame')
                break
            
            frame_num += 1

            # Perform detection on the current frame
            results = model(frame, conf=min_conf, classes=[0, 1, 2, 3, 4, 5, 6, 7])

            # Iterate over the results (detection boxes)
            for result in results: 
                # Extract the bounding box, confidence, and class
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cid = int(box.cls[0])



                    # Write the detection to the text file
                    output_file.write(f"{cid} ")
            output_file.write("\n")    



            # Optional: Display the frame (with bounding boxes)
            # cv2.imshow("Frame", frame)

            # Exit on 'q' key press (optional)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cap.release()
        # cv2.destroyAllWindows()

    print(f"Detection results saved to {output_txt_path}")



if __name__ == "__main__":
    
    
    
    yolovTextgen(INPUT_VIDEO,OUTPUT_TXT )
        # process_video(f'./encoded_videos/{video}', f'./yolo_txt_files/{video_name}.txt')