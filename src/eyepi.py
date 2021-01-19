
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import boto3
import json
import datetime
import concurrent.futures
import traceback
from collections import deque
import videostream
from videostream import VideoStream
from objectdetector import EyePiDetectionEvent, EyePiObjectDetector
from recorder import EyePiRecordingEvent, EyePiRecorder

"""
Eyepi - object detection on Raspberry Pi with Tensorflow + notification via AWS cloud. 
"""


def main(args):

    """
    Main entry point after parsing args
    """

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    LOG_LEVEL = args.loglevel
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu
    s3bucket_name = args.s3bucket

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Have to do a weird fix for label map if using the COCO "starter model" from
    # https://www.tensorflow.org/lite/models/object_detection/overview
    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(
        inputdevice=args.inputdevice,
        resolution=(imW,imH),
        framerate=30,
    )
    videostream.start()
    time.sleep(1)

    # Create the object that will process video frames and recognition results and upload to s3 and send alerts
    eyePiObjectDetector = EyePiObjectDetector(
        labels=labels,
        s3bucket_name=s3bucket_name,
        target_object=args.targetobject,
        min_conf_threshold=min_conf_threshold,
    )

    recorder_state = 'IDLE'
    if int(args.recordxseconds) > 0:
        recorder_state = 'RECORD_ON_NEXT_EVENT'
    eyePiRecorder = EyePiRecorder(
        initial_state = recorder_state,
        recording_length_seconds = int(args.recordxseconds),
        s3bucket_name=s3bucket_name,
    )

    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        if frame1 is None:
            print("Videostream returned empty frame, waiting ..")
            time.sleep(0.01)
            continue

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                if LOG_LEVEL == 'DEBUG':
                    print(f"Detected {label}")

                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        print("Frame rate: {}".format(frame_rate_calc))

        # Process EyePiDetectionEvent
        eyePiEvent = EyePiDetectionEvent(
            frame=frame,
            detected_classes=classes,
            detected_scores=scores
        )
        eyePiObjectDetector.process_event(event=eyePiEvent)

        # Send to recorder if it's enabled
        eyePiRecordingEvent = EyePiRecordingEvent(
            frame=frame,
        )
        eyePiRecorder.process_event(event=eyePiRecordingEvent)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    videostream.stop()
    eyePiObjectDetector.shutdown()
    eyePiRecorder.shutdown()





if __name__ == "__main__":

    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3bucket', help='The name of the s3 bucket where capture videos should be uploaded to',
                        required=True)
    parser.add_argument('--targetobject', help='The object to detect.  Eg, person',
                        default='person')
    parser.add_argument('--loglevel', help='The loglevel to use, INFO, DEBUG',
                        default='INFO')
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        default='modeldir')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.72)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')
    parser.add_argument('--recordxseconds', help='Record video with detection overlay for X seconds and upload to S3 and send alert',
                        default='0')
    parser.add_argument('--inputdevice',
                        help='The input source device or AVI file to get video input.  Defaults to camera at /dev/video0',
                        default='/dev/video0')

    args = parser.parse_args()
    main(args)
