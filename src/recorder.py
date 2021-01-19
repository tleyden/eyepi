from util import push_event_to_s3
from util import future_callback_error_logger
import cv2
import boto3
import datetime
import concurrent.futures
import videostream

class EyePiRecordingEvent(object):
    """
    An event for the EyePiRecorder to possibly record
    """
    def __init__(self, frame):
        self.frame = frame

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

        print(f"Recording for {self.recording_length_seconds} seconds")
        self.state = 'RECORDING'
        self.start_recording_timestamp = datetime.datetime.utcnow().timestamp()

        # The AVI file extension matters a lot - See https://stackoverflow.com/questions/30509573/writing-an-mp4-video-using-python-opencv
        self.latest_recording_file_name = "recording_{}.avi".format(datetime.datetime.utcnow().timestamp())
        self.latest_recording_file_path = "/tmp/{}".format(self.latest_recording_file_name)
        (h, w) = event.frame.shape[:2]

        # The FPS of the writer which sets the playback speed.  It should be as close to the
        # recording FPS as possible so that wall clock time passage of recording matches up with
        # wall clock time passage of playback.  How do we know the recording FPS though?
        fps = 4.5  # Set to the average FPS on Raspi 4

        self.writer = cv2.VideoWriter(
            self.latest_recording_file_path,
            videostream.fourcc,
            fps,
            (w, h),
            True,
        )

    def transition_to_idle_state(self):

        print("Finished capturing recording, returning to IDLE state")

        self.state = 'IDLE'
        print("Finished recording video: {}".format(self.latest_recording_file_path))

        self.writer.release()

        self.start_recording_timestamp = None

    def send_alert_with_recording(self):

        push_event_to_s3(
            self.s3_client,
            self.s3bucket_name,
            self.latest_recording_file_path,
            self.latest_recording_file_name,
            "Recording",
            1.0)