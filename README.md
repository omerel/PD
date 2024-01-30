# Person Detection Using YOLO and Hugging Face Transformers

This project is a Python script for real-time person detection using a webcam. It utilizes the YOLO (You Only Look Once) model for object detection and the Hugging Face Transformers library. The script captures video from the webcam, identifies persons in the video frame, and logs the detection count at specified intervals.

## Getting Started

### Dependencies

Ensure you have the following installed:

- Python 3.x
- OpenCV
- PyTorch
- Transformers (Hugging Face)
- A webcam or a video capture device

### Installation

Install the required packages:
```bash
pip install -r requirements.txt
```
### Usage
Run the script from the command line. You can specify parameters like the score threshold for detection, log interval in seconds, and the log output path:
```bash
python main.py --score_threshold 0.9 --log_interval 1 --log_output_path './logs/'
```
Press 'q' to quit the webcam feed.

### Features
- Real-time person detection using webcam
- Adjustable score threshold for detection
- Logging the count of persons detected at regular  intervals
- Customizable log file output
