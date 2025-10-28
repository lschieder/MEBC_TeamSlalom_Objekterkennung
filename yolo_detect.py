import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO


# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), or index of Picamera ("picamera0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--output', help='Name of output video file (default: "output_with_detections.avi")',
                    default='output_with_detections.avi')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
output_filename = args.output

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': 
        cap_arg = img_source
        cap = cv2.VideoCapture(cap_arg)
    elif source_type == 'usb': 
        cap_arg = usb_idx
        cap = cv2.VideoCapture(cap_arg, cv2.CAP_DSHOW)  # DirectShow für bessere OBS Virtual Camera Kompatibilität

    # Get video properties for output video
    if not user_res:
        resW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        resize = False
    else:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    if not user_res:
        print('ERROR: Must specify --resolution for Picamera.')
        sys.exit(0)
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Set up video writer for video/camera sources
video_writer = None
if source_type in ['video', 'usb', 'picamera']:
    # Get FPS from source (for video) or use default (for cameras)
    if source_type == 'video':
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0
    else:
        fps = 30.0
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (resW, resH))
    print(f'Video output will be saved to: {output_filename}')

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Window name for consistent reference
window_name = 'YOLO detection results'

# Begin inference loop
try:
    while True:

        t_start = time.perf_counter()

        # Load frame from image source
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images have been processed. Exiting program.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count = img_count + 1
        
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('Reached end of the video file. Exiting program.')
                break
        
        elif source_type == 'usb':
            ret, frame = cap.read()
            if (frame is None) or (not ret):
                print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
                break

        elif source_type == 'picamera':
            frame = cap.capture_array()
            if (frame is None):
                print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
                break

        # Resize frame to desired display resolution
        if resize == True:
            frame = cv2.resize(frame,(resW,resH))

        # Run inference on frame
        results = model(frame, verbose=False)

        # Extract results
        detections = results[0].boxes

        # Initialize variable for basic object counting example
        object_count = 0

        # Go through each detection and get bbox coords, confidence, and class
        for i in range(len(detections)):

            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:

                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Basic example: count the number of objects in the image
                object_count = object_count + 1

        # Calculate and draw framerate (if using video, USB, or Picamera source)
        if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        
        # Display detection results
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.imshow(window_name, frame)
        
        # Write frame to output video if applicable
        if video_writer is not None:
            video_writer.write(frame)

        # Check if window was closed by clicking X
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print('Window closed by user. Exiting program.')
            break

        # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
        if source_type == 'image' or source_type == 'folder':
            key = cv2.waitKey()
        elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
            key = cv2.waitKey(5)
        
        if key == ord('q') or key == ord('Q'):
            print('Quit key pressed. Exiting program.')
            break
        elif key == ord('s') or key == ord('S'):
            cv2.waitKey()
        elif key == ord('p') or key == ord('P'):
            cv2.imwrite('capture.png',frame)
            print('Screenshot saved as capture.png')
        
        # Calculate FPS for this frame
        t_stop = time.perf_counter()
        frame_rate_calc = float(1/(t_stop - t_start))

        # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
        if len(frame_rate_buffer) >= fps_avg_len:
            temp = frame_rate_buffer.pop(0)
            frame_rate_buffer.append(frame_rate_calc)
        else:
            frame_rate_buffer.append(frame_rate_calc)

        # Calculate average FPS for past frames
        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print('\nProgram interrupted by user (Ctrl+C).')

finally:
    # Clean up - this block always executes
    print(f'\nAverage pipeline FPS: {avg_frame_rate:.2f}')
    
    # Release video capture
    if source_type == 'video' or source_type == 'usb':
        cap.release()
        print('Video capture released.')
    elif source_type == 'picamera':
        cap.stop()
        print('Picamera stopped.')
    
    # Release video writer
    if video_writer is not None:
        video_writer.release()
        print(f'Video saved successfully: {output_filename}')
    
    # Destroy all OpenCV windows
    cv2.destroyAllWindows()
    print('All windows closed. Program terminated cleanly.')
