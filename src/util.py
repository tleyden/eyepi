import json
import traceback

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


def future_callback_error_logger(future):
    """
    Utility to help log the result or exception of a future so it doesn't get lost in the stack
    """
    try:
        result = future.result()
    except Exception as e:
        print("Executor Exception: {}".format(e))
        traceback.print_exc()