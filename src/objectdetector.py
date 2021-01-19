from util import push_event_to_s3
from util import future_callback_error_logger
import cv2
import boto3
import datetime
import concurrent.futures
import videostream

class EyePiDetectionEvent(object):
    """
    An event correlated with objects being detected in the video stream
    """
    def __init__(self, frame, detected_classes, detected_scores):
        self.frame = frame
        self.detected_classes = detected_classes
        self.detected_scores = detected_scores


class EyePiObjectDetector(object):
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
        future = self.executor.submit(
            self.process_event_async,
            event=event,
        )
        future.add_done_callback(future_callback_error_logger)

    def process_event_async(self, event):
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
        self.writer.write(event.frame)
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
            videostream.fourcc,
            fps,
            (w, h),
            True,
        )

    def transition_to_idle_state(self, event):

        print("Finished capturing video, returning to IDLE state")

        self.state = 'IDLE'
        print("Finished capturing video: {}".format(self.latest_capture_file_path))

        # Save video and json with signed s3 url and push both to s3
        filename = self.latest_capture_file_path
        object_name = self.latest_capture_file_name

        self.writer.release()
        self.num_captured_frames = 0

        push_event_to_s3(
            s3_client=self.s3_client,
            bucket_name=self.bucket_name,
            filename=filename,
            object_name=object_name,
            detected_object=self.target_object,
            detection_confidence=float(self.last_object_detected_confidence),
        )

