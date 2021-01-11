
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

"""
Eyepi - object detection on Raspberry Pi with Tensorflow + notification via AWS cloud. 
"""

# Global constants

# This video codec worked after some trial and error as long as the filename was ".avi"
# See https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
fourcc = cv2.VideoWriter_fourcc(*'MJPG')

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
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    # Create the object that will process video frames and recognition results and upload to s3 and send alerts
    eyePiEventStream = EyePiEventStream(
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
        recording_length_seconds = args.recordxseconds,
        s3bucket_name=s3bucket_name,
    )

    while True:

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

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

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        print("Frame rate: {}".format(frame_rate_calc))

        # Process EyePiDetectionEvent
        eyePiEvent = EyePiDetectionEvent(
            frame=frame,
            detected_classes=classes,
            detected_scores=scores
        )
        eyePiEventStream.process_event(eyePiEvent)

        # Send to recorder if it's enabled
        eyePiRecorder.process_event(frame=frame)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    videostream.stop()
    eyePiEventStream.shutdown()
    eyePiRecorder.shutdown()

class VideoStream:
    """
    Define VideoStream class to handle streaming of video from webcam in separate processing thread
    """
    def __init__(self,resolution=(640,480),framerate=30):

        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, fourcc)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


class EyePiRecorder(object):
    """
    Recorder used for things like positioning the camera and seeing how well the model works.

    After it's finished recording, it will:
     - Upload the recording to S3
     - Send email alert

    This is for easy viewing from a mobile device.

    initial_state: 'IDLE' or 'RECORDING'
    recording_length_seconds: How long in seconds that recordings should last when triggered
    s3bucket_name: the target s3 bucket where video clips should be written

    """
    def __init__(self, initial_state, recording_length_seconds, s3bucket_name):

        if initial_state not in ['IDLE', 'RECORDING', 'RECORD_ON_NEXT_EVENT']:
            raise Exception("Unexpected state")

        self.start_recording_timestamp = None
        self.recording_length_seconds = recording_length_seconds
        self.s3bucket_name = s3bucket_name

        # State machine
        self.state = initial_state

        # Threadpool executor to keep from blocking the main thread.
        # Keep max workers at 1 so it's easier to reason about concurrent access to data.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Create the s3 client
        self.s3_client = boto3.client('s3')

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def process_event_async(self, event):
        """
        States:

            IDLE
            RECORDING
            RECORD_ON_NEXT_EVENT
        """

        if self.state == 'IDLE':
            self.process_event_idle_state(event)
        elif self.state == 'RECORDING':
            self.process_event_recording_state(event)
        elif self.state == 'RECORD_ON_NEXT_EVENT':
            self.transition_to_recording_state(event)
            self.process_event_recording_state(event)
        else:
            raise Exception("Unknown state: {}".format(self.state))

    def process_event(self, event):
        future = self.executor.submit(
            self.process_event_async,
            event=event,
        )
        future.add_done_callback(future_callback_error_logger)

    def process_event_idle_state(self, event):
        # Ignore any events while not recording
        pass

    def process_event_recording_state(self, event):

        # Write event to output file
        self.writer.write(event.frame)

        # Check if we're done recording, if so then transition to idle state and send alert
        now = datetime.datetime.utcnow().timestamp()
        delta_seconds = now - self.start_recording_timestamp
        if delta_seconds >= self.recording_length_seconds:
            self.transition_to_idle_state()
            self.send_alert_with_recording()

    def transition_to_recording_state(self, event):

        self.state = 'RECORDING'
        self.start_recording_timestamp = datetime.datetime.utcnow().timestamp()

        # The AVI file extension matters a lot - See https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
        self.latest_recording_file_name = "recording_{}.avi".format(datetime.datetime.utcnow().timestamp())
        self.latest_recording_file_path = "/tmp/{}".format(self.latest_recording_file_name)
        (h, w) = event.frame.shape[:2]

        # The FPS of the writer.  I guess this sets the playback speed?
        # What should this be set to?
        fps = 1

        self.writer = cv2.VideoWriter(
            self.latest_recording_file_path,
            fourcc,
            fps,
            (w, h),
            True,
        )

    def transition_to_idle_state(self):

        print("Finished capturing video, returning to IDLE state")

        self.state = 'IDLE'
        print("Finished recording video: {}".format(self.latest_recording_file_path))

        self.writer.release()

        self.start_recording_timestamp = None

    def send_alert_with_recording(self):

        push_event_to_s3(
            self.s3_client,
            self.bucket_name,
            self.latest_recording_file_path,
            self.latest_recording_file_name,
            "Recording",
            1.0)

class EyePiDetectionEvent(object):
    """
    An event correlated with objects being detected in the video stream
    """
    def __init__(self, frame, detected_classes, detected_scores):
        self.frame = frame
        self.detected_classes = detected_classes
        self.detected_scores = detected_scores


class EyePiEventStream(object):
    """
    Handles overall EyePi functionality in terms of injesting event stream from
    camera and model

    labels: the full list of labels known by the model
    s3bucket_name: the target s3 bucket where video clips should be written
    target_object: the object that should be alerted on, model-dependent and should be present in labels
    min_conf_threshold: minimum confidence threshold for detection, a number between 0.0 - 1.0
    """
    def __init__(self, labels, s3bucket_name, target_object, min_conf_threshold):
        self.labels = labels
        self.bucket_name = s3bucket_name
        self.target_object = target_object  # TODO: validate that the target_object is present in labels
        self.min_conf_threshold = min_conf_threshold
        self.num_captured_frames = 0

        # The length of the capture is determined by a frame count rather than time-based
        # Since we're setting the output videowriter to 1 FPS, presumably this will
        # translate into 10 seconds of video
        self.num_frames_per_video = 10

        # State machine
        self.state = 'IDLE'
        self.last_object_detected_confidence = float(0)

        # Threadpool executor to keep from blocking the main thread.
        # Keep max workers at 1 so it's easier to reason about concurrent access to data.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Create the s3 client and validate the creds by listing buckets
        self.s3_client = boto3.client('s3')
        self.validate_s3_creds()


    def validate_s3_creds(self):
        """
        Fail-fast and make sure we can at least list the s3 buckets
        """
        self.s3_client.list_buckets()
        print("S3 creds have been verified")

    def shutdown(self):
        self.executor.shutdown(wait=True)

    def process_event(self, event):
        """

        After the target object is detected with a high enough threshold (eg, 72%), it will
        enter the OBJECT_DETECTED_CAPTURING_VIDEO state and:

        - Capture each video frame to a buffer in memory for the next 5 seconds
        - Trigger the state transition back to the idle state, which will do the following:
            - Kick off a separate thread so we don't block the VideoStream thread
            - Collect the frames from the buffer and write to a file
            - Upload the video file to s3 (alert_123.mp4)
            - Generate a pre-signed url for the video file
            - Upload an alert_123.txt file like "A person was detected with 72% accuracy: {presigned_url} to s3

        States:

            IDLE
            OBJECT_DETECTED_CAPTURING_VIDEO

        """

        if self.state == 'IDLE':
            self.process_event_idle_state(event)
        elif self.state == 'OBJECT_DETECTED_CAPTURING_VIDEO':
            self.process_event_capturing_state(event)
        else:
            raise Exception("Unknown state: {}".format(self.state))

    def object_detected(self, event):
        found_object = False
        # Loop over all scores, find the corresponding object name, and see if it's the object we care about
        for i in range(len(event.detected_scores)):
            if ((event.detected_scores[i] > self.min_conf_threshold) and (event.detected_scores[i] <= 1.0)):
                object_name = self.labels[int(event.detected_classes[i])]
                if object_name.lower() == self.target_object.lower():
                    found_object = True
                    self.last_object_detected_confidence = event.detected_scores[i]
                    break

        return found_object

    def process_event_idle_state(self, event):
        if self.object_detected(event) == True:
            self.transition_to_capturing_state(event)

    def process_event_capturing_state(self, event):

        self.num_captured_frames += 1
        self.writer.write(event.frame)  # TODO: this should happen in self.executor() so it doesn't block the main thread
        print("Captured frame {}/{}".format(self.num_captured_frames, self.num_frames_per_video))

        if self.num_captured_frames > self.num_frames_per_video:
            self.transition_to_idle_state(event)

    def transition_to_capturing_state(self, event):

        print(f"{self.target_object} detected!!  Capturing video")

        self.state = 'OBJECT_DETECTED_CAPTURING_VIDEO'
        self.num_captured_frames = 0

        # The AVI file extension matters a lot - See https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
        self.latest_capture_file_name = "alert_{}.avi".format(datetime.datetime.utcnow().timestamp())
        self.latest_capture_file_path = "/tmp/{}".format(self.latest_capture_file_name)
        (h, w) = event.frame.shape[:2]

        # The FPS of the writer.  I guess this sets the playback speed?
        # What should this be set to?
        fps = 1

        self.writer = cv2.VideoWriter(
            self.latest_capture_file_path,
            fourcc,
            fps,
            (w, h),
            True,
        )

    def transition_to_idle_state(self, event):

        print("Finished capturing video, returning to IDLE state")

        self.state = 'IDLE'
        print("Finished capturing video: {}".format(self.latest_capture_file_path))

        # Kick off async task to save video and json with signed s3 url and push both to s3
        future = self.executor.submit(
            self.push_event_to_s3,
            event=event,
            filename=self.latest_capture_file_path,
            object_name=self.latest_capture_file_name,
        )
        future.add_done_callback(future_callback_error_logger)

        self.writer.release()  # TODO: should happen in self.executor() task
        self.num_captured_frames = 0

    def push_event_to_s3(self, event, filename, object_name):
        """
        - Push video to s3
        - Generate signed URL for video
        - Write an alert file that says "Person detected .. <link to video>"
        - Write alert file to s3
        """

        try:
            print("Uploading {} -> {}/{} .. ".format(filename, self.bucket_name, object_name))
            response = self.s3_client.upload_file(
                filename,
                self.bucket_name,
                object_name,
            )
            print("Finished uploading {} -> {}/{} .. ".format(filename, self.bucket_name, object_name))

            # Make the video capture file public
            # TODO: use signed URLs instead of making the file public
            self.s3_client.put_object_acl(ACL='public-read', Bucket=self.bucket_name, Key="%s" % (object_name))


            # Create and upload alert meta file
            public_url = f'https://{self.bucket_name}.s3.amazonaws.com/{object_name}'

            alert_meta = {
                "detected_object": self.target_object,
                "detection_confidence": float(self.last_object_detected_confidence),
                "captured_video_url": public_url,
            }

            alert_meta_object_name = "{}.json".format(object_name)
            alert_meta_filepath = "/tmp/{}".format(alert_meta_object_name)
            f = open(alert_meta_filepath, "a")
            f.write(json.dumps(alert_meta))
            f.close()

            print("Uploading {} -> {}/{} .. ".format(alert_meta_filepath, self.bucket_name, alert_meta_object_name))

            response = self.s3_client.upload_file(
                alert_meta_filepath,
                self.bucket_name,
                alert_meta_object_name,
            )
            print("Finished uploading {} -> {}/{} .. ".format(alert_meta_filepath, self.bucket_name, alert_meta_object_name))

        except Exception as e:
            print("Exception writing {} to s3: {}".format(object_name, str(e)))
            raise e

def future_callback_error_logger(future):
    """
    Utility to help log the result or exception of a future so it doesn't get lost in the stack
    """
    try:
        result = future.result()
    except Exception as e:
        print("Executor Exception: {}".format(e))

def push_event_to_s3(s3_client, bucket_name, filename, object_name, detected_object, detection_confidence):
    """

    TODO: replace other push_event_to_s3 with call to this one

    - Push video to s3
    - Generate signed URL for video
    - Write an alert file that says "Person detected .. <link to video>"
    - Write alert file to s3
    """

    try:
        print("Uploading {} -> {}/{} .. ".format(filename, bucket_name, object_name))
        response = s3_client.upload_file(
            filename,
            bucket_name,
            object_name,
        )
        print("Finished uploading {} -> {}/{} .. ".format(filename, bucket_name, object_name))

        # Make the video capture file public
        # TODO: use signed URLs instead of making the file public
        s3_client.put_object_acl(ACL='public-read', Bucket=bucket_name, Key="%s" % (object_name))


        # Create and upload alert meta file
        public_url = f'https://{bucket_name}.s3.amazonaws.com/{object_name}'

        alert_meta = {
            "detected_object": detected_object,
            "detection_confidence": detection_confidence,
            "captured_video_url": public_url,
        }

        alert_meta_object_name = "{}.json".format(object_name)
        alert_meta_filepath = "/tmp/{}".format(alert_meta_object_name)
        f = open(alert_meta_filepath, "a")
        f.write(json.dumps(alert_meta))
        f.close()

        print("Uploading {} -> {}/{} .. ".format(alert_meta_filepath, bucket_name, alert_meta_object_name))

        response = s3_client.upload_file(
            alert_meta_filepath,
            bucket_name,
            alert_meta_object_name,
        )
        print("Finished uploading {} -> {}/{} .. ".format(alert_meta_filepath, bucket_name, alert_meta_object_name))

    except Exception as e:
        print("Exception writing {} to s3: {}".format(object_name, str(e)))
        raise e

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

    args = parser.parse_args()
    main(args)