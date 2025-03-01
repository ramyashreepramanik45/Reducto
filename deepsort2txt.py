import cv2
import pandas as pd
from ultralytics import YOLO



# Path to store the output text file containing tracking data
OUTPUT_PATH = "mp4_deepsort_txt/1.txt"

# Path to input video file
VIDEO_PATH = "videos/1.mp4"

# Load the YOLO model (pre-trained on the COCO dataset)
model = YOLO("yolov8n.pt") 

TRANSPORT_CLASSES = ["car", "truck", "bus", "motorbike", "bicycle"]

def process_video(video_path, output_txt = OUTPUT_PATH ) :
    
    """
    Processes a video to detect and track transport objects using YOLO and ByteTrack.
    
    Args:
        video_path (str): Path to the input video file.
        output_txt (str): Path to the output text file where results will be saved.
    """
    
    
    cap = cv2.VideoCapture(video_path)
    frame_no = -1

   
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        # Perform object detection and tracking
        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")  
        # Extract detected bounding boxes
        detections = results[0].boxes

        with open(output_txt, "a") as file:
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                obj_id = int(box.id[0].item()) if box.id is not None else -1  
                cls = int(box.cls[0].item())  
                class_name = model.names[cls]

                
                if class_name in TRANSPORT_CLASSES:
                    file.write(f"{frame_no},{obj_id},{class_name},{x1},{y1},{x2-x1},{y2-y1}\n")

                    
                    color = (0, 255, 0) 
                    label = f"{class_name} {obj_id}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

       
        cv2.imshow("Tracking", frame)

       
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




def fill_missing_frames(csv_path, output_path):
    """
    This function fills in missing frames in the tracking data by duplicating the last available frame's data.
    
    When an object disappears temporarily due to occlusion or tracking failures, this function ensures that the
    object remains present in every frame by propagating the last known state until a new detection appears.
    
    Args:
        csv_path (str): Path to the input CSV file containing tracking data.
        output_path (str): Path to the output CSV file with missing frames filled.
    """
    
    df = pd.read_csv(csv_path, header=None)

    df.columns = ["frame", "object_id", "class", "x", "y", "w", "h"]

  
    df = df.sort_values(by="frame").reset_index(drop=True)

    min_frame, max_frame = df["frame"].min(), df["frame"].max()
    existing_frames = set(df["frame"])

    filled_data = []
    last_valid_row = None

    for frame in range(min_frame, max_frame + 1):
        if frame in existing_frames:

            last_valid_row = df[df["frame"] == frame]
            filled_data.extend(last_valid_row.values.tolist())
        else:
           
            if last_valid_row is not None:
                dummy_row = last_valid_row.iloc[-1].copy()
                dummy_row["frame"] = frame  
                filled_data.append(dummy_row.values.tolist())

    
    filled_df = pd.DataFrame(filled_data, columns=df.columns)

    # Save the updated tracking data to CSV
    filled_df.to_csv(output_path, index=False, header=False)
    print(f"Missing frames filled and saved to {output_path}")


if __name__ == "__main__":
    video_path = VIDEO_PATH
    
    # Run object detection and tracking
    process_video(video_path)
    # Fill missing frames in tracking data
    fill_missing_frames(OUTPUT_PATH,OUTPUT_PATH)
