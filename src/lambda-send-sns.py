import json
import urllib.parse
import boto3
import os

SNS_TOPIC_ARN = os.environ['SNS_TOPIC_ARN']
s3 = boto3.client('s3')

def lambda_handler(event, context):
    
    print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    # Ignore all files unless they end with .json
    if not key.endswith(".json"):
        print("Ignoring file since it's not a json file")
        return 
    
    event_json = {}
    
    try:
        print("Parsing json file")
        response = s3.get_object(Bucket=bucket, Key=key)
        bodystr = response['Body'].read()  # should it also .decode('utf-8')?
        event_json = json.loads(bodystr)
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
        
    try:
        sns = boto3.client('sns')
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject='EyePi object detected',
            Message='{} detected with {}% confidence.  Watch captured video: {}'.format(
                event_json['detected_object'],
                event_json['detection_confidence'],
                event_json['captured_video_url'],
            )
        )
        
    except Exception as e:
        print(e)
        print('Error sending SNS')
        raise e
