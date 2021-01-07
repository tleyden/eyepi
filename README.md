# eyepi

Security camera object detection software that runs on a Raspberry Pi.

When a person is detected by the model it sends an email/text notification with a 5 second video clip stored on AWS S3.

![image](https://user-images.githubusercontent.com/296876/103612582-9e8f6700-4ed9-11eb-9266-f7e5ec927d9e.png)

Table of Contents
=================

   * [eyepi](#eyepi)
      * [Requirements](#requirements)
      * [Setup Raspberry Pi (Headless)](#setup-raspberry-pi-headless)
      * [Install Eyepi](#install-eyepi)
         * [Clone repo](#clone-repo)
         * [Install docker engine](#install-docker-engine)
         * [Build Eyepi Docker image from Dockerfile](#build-eyepi-docker-image-from-dockerfile)
         * [Setup AWS Cloud](#setup-aws-cloud)
            * [IAM User with S3FullAccess](#iam-user-with-s3fullaccess)
            * [S3 Bucket](#s3-bucket)
            * [SNS topic + subscription](#sns-topic--subscription)
            * [Lambda function](#lambda-function)
         * [Run eyepi docker container](#run-eyepi-docker-container)
      * [Test it out](#test-it-out)
      * [Run as a daemon process in the background](#run-as-a-daemon-process-in-the-background)
      * [References](#references)
         
## Requirements

* Raspberry Pi 3 or 4
* Raspbian Buster OS (Raspbian GNU/Linux 10)
* AWS Account

## Setup Raspberry Pi (Headless)

* Download *Raspberry Pi OS with desktop and recommended software* from the [Raspi Downloads](https://www.raspberrypi.org/software/operating-systems/) page
* Burn the image to a microsd card using Balena Etcher using [these instructions](https://youtu.be/PijX8GDco-g)
* Remove and re-insert the sd card reader to your mac to force it to mound the sd card
* Add an empty `ssh` file in the root of the mounted sd card.  See [Enable SSH on a headless Raspberry Pi (add file to SD card on another machine)](https://www.raspberrypi.org/documentation/remote-access/ssh/)
* Tell it how to connect to your wifi by adding a `wpa_supplicant.conf` file -- see [Setting up wireless networking
](https://www.raspberrypi.org/documentation/configuration/wireless/headless.md) 
* Unmount sd card and insert into pi, and boot the pi
* [Find the IP address of your pi](https://www.raspberrypi.org/documentation/remote-access/ip-address.md)
* ssh into your pi with username: `pi` password: `raspberry` -- see [Raspberry Pi forum post](https://www.raspberrypi.org/forums/viewtopic.php?t=173195)
* [Enable the camera](https://www.raspberrypi.org/documentation/configuration/camera.md) - requires restart

## Install Eyepi

### Clone repo

On the raspberry pi, `git clone` this repo.

```
$ git clone git@github.com:tleyden/eyepi.git
$ cd eyepi
```

### Install docker engine

Install docker engine using [the convenience script](https://docs.docker.com/engine/install/debian/#install-using-the-convenience-script).

Add the pi user to the docker group:

```
$ sudo usermod -aG docker pi
```

### Build Eyepi Docker image from Dockerfile

```
$ docker build . -t eyepi
```

Make sure you can run the docker image:

```
$ docker run -it --device=/dev/video0:/dev/video0 eyepi eyepi.py --s3bucket eyepi
```

At this point you should see an error: `botocore.exceptions.NoCredentialsError: Unable to locate credentials` since you aren't passing in the AWS creds yet.  That's covered in a later step.

### Setup AWS Cloud

#### IAM User with S3FullAccess

Create a new IAM User called "eyepi" and associate S3FullAccess permissions.  (TODO: limit permissions)

![Screen Shot 2021-01-06 at 10 20 29 PM](https://user-images.githubusercontent.com/296876/103858698-ab44c400-506d-11eb-8957-e7db7d86951b.png)

* Disallow web console access
* Generate an access key / secret key pair and store these somewhere secure, you will need in a later step

#### S3 Bucket

Create a new bucket with a globally unique name - for example `<your-email>-eyepi`, and settings:

Step 1: Allow public access

![Screen Shot 2021-01-06 at 10 25 12 PM](https://user-images.githubusercontent.com/296876/103858953-15f5ff80-506e-11eb-9cf5-96ebecfdc40e.png)

Step 2: Use the default ACL 

![Screen Shot 2021-01-06 at 10 25 20 PM](https://user-images.githubusercontent.com/296876/103858949-14c4d280-506e-11eb-81bf-1a360715f114.png)

#### SNS topic + subscription

Create a new SNS topic called `EyePiNotificationsSNS`, and create an email subscription with your email address.

![Screen Shot 2021-01-06 at 10 28 29 PM](https://user-images.githubusercontent.com/296876/103859403-d845a680-506e-11eb-8e08-355dd107a1e3.png)

#### Lambda function

Step 1: Create a new lambda function from the s3 bucket blueprint

![Screen Shot 2021-01-06 at 10 32 03 PM](https://user-images.githubusercontent.com/296876/103859815-8ea98b80-506f-11eb-840a-a8649f2cb5f1.png)

Step 2: Set function name and role name to EyepiLambda, choose your eyepi s3 bucket for the s3 trigger

![Screen Shot 2021-01-06 at 10 33 06 PM](https://user-images.githubusercontent.com/296876/103859922-bd276680-506f-11eb-8fd5-92a0c998ce2d.png)

Step 3: Copy the code from `lambda-send-sns.py` from this repo into the lambda code text area, and hit the "Create" button

![Screen Shot 2021-01-06 at 10 33 18 PM](https://user-images.githubusercontent.com/296876/103860004-e1834300-506f-11eb-95a0-cdf934a7161f.png)

Step 4. Go to the lambda function permission section and click the role

![Screen Shot 2021-01-06 at 10 33 51 PM](https://user-images.githubusercontent.com/296876/103860173-24451b00-5070-11eb-9bbc-5e3a6fc744a6.png)

Step 5: Add the `AmazonS3ReadOnlyAccess` and `AmazonSNSFullAccess` policies

![Screen Shot 2021-01-06 at 10 34 11 PM](https://user-images.githubusercontent.com/296876/103860137-17282c00-5070-11eb-98ac-a62ed4dd7549.png)

Step 6: In the lambda configuration section, add the SNS ARN as an environment variable called `SNS_TOPIC_ARN`

![Screen Shot 2021-01-06 at 10 41 57 PM](https://user-images.githubusercontent.com/296876/103860792-19d75100-5071-11eb-9253-6036c35d53a4.png)

### Run eyepi docker container

First set the boto env variables using the access key and secret key for the IAM user created above (without the `{}`'s):

````
$ export AWS_ACCESS_KEY_ID="{your-aws-access-key}"
$ export AWS_SECRET_ACCESS_KEY="{your-secret-key}"
````

Now launch the docker container

```
$ docker run -it -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY --device=/dev/video0:/dev/video0 eyepi eyepi.py --s3bucket <your-bucket-name>
```

## Test it out

You should see output like:

```
S3 creds OK
read a frame
Frame rate: 1
read a frame
Frame rate: 4.521088120478825
```

Now walk in front of the camera for a few seconds, and you should see this output:

```
Person detected!!  Capturing video
read a frame
...
Captured frame 6/5
Finished capturing video, returning to IDLE state
Finished capturing video: /tmp/alert_1610002882.726963.avi
Uploading /tmp/alert_1610002882.726963.avi -> eyepi/alert_1610002882.726963.avi ..
Finished uploading /tmp/alert_1610002882.726963.avi -> eyepi/alert_1610002882.726963.avi ..
```

and you should receive an email alert with subject "EyePi person detected" and text:

> person detected with 0.73828125% confidence.  Watch captured video: https://<your-bucket>.s3.amazonaws.com/alert_1610002882.726963.avi

Clicking the link on iOS Safari should play it directly.

## Run as a daemon process in the background

Kill the docker container previously launched, and re-run and replace the arguments:

```
-it
```

with:

```
-itd
```

Where the `-d` tells docker to daemonize the process.  It will show the container id, eg: `ce43d7b59c`, and you can view the logs with:

```
$ docker logs -f ce43d7b59c
```

## References

* https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
* https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html
