import os
import cv2
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
from datetime import datetime
import time
import csv
import argparse
import shutil


# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description='Person Detection Script')
    parser.add_argument('--score_threshold', type=float, default=0.9, help='Score threshold for detection')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval in seconds')
    parser.add_argument('--log_output_path', type=str, default='./logs/', help='Log output path')
    return parser.parse_args()

# prediction function
def predict(frame,model,image_processor):

    inputs = image_processor(images=frame, return_tensors="pt")
    outputs = model(**inputs)
        
    # print results
    frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_sizes = torch.tensor([frame_g.shape])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]   
    return results


def main(args):

    # copy transformer model into cache if not exsist
    hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
    )
    model_cache_path = os.path.join(hf_cache_home, "hub","models--hustvl--yolos-tiny")
    if not os.path.exists(model_cache_path):
        # Copy the entire directory
        shutil.copytree("./setup/models--hustvl--yolos-tiny", model_cache_path)
        
    # initialize arguments
    SCORE_THRESHOLD = args.score_threshold
    LOG_INTERVAL_IN_SECONDS = args.log_interval
    LOG_OUTPUT_PATH = args.log_output_path
    
    # Initialize YOLO model and processor
    model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
    
    # Capture video from webcam (use 0 for default camera)
    cap = cv2.VideoCapture(0)

    # initialize last second
    last_second = -1

    filename = f"{LOG_OUTPUT_PATH}log_preson_detected_{str(datetime.now().isoformat())[:19].replace(':','_')}.csv"
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the header
        csvwriter.writerow(['Timestamp','N_Perosns_Detected'])
        csvfile.close()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = datetime.now()
        if current_time.second - last_second >= LOG_INTERVAL_IN_SECONDS:
            last_second = current_time.second
        
        # Predict frame
        results = predict(frame,model,image_processor) 
        # Count persons in the frame and draw box
        person_count = 0
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            if (model.config.id2label[label.item()] == 'person') & (score.item() > SCORE_THRESHOLD): 
                person_count+=1
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(frame, f'{round(score.item(), 3)}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
        # Display the frame with person count
        cv2.putText(frame, f'Persons: {person_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Live Frame', frame)
        
        # Record in log file
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Writing the header
            csvwriter.writerow([current_time,person_count])
            print(current_time,person_count)
            csvfile.close()
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csvfile.close()
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()  # Parse arguments
    main(args)