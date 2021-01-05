# eyepi

Security camera software that runs on Raspberry Pi with people detection

When a person is detected by the model send a notification with a 5 second video clip stored on AWS S3.

Send at most one notification per 5 minute period.

![image](https://user-images.githubusercontent.com/296876/103612582-9e8f6700-4ed9-11eb-9266-f7e5ec927d9e.png)


## Requirements

* Raspberry Pi 3 or 4
* Raspbian Buster OS
* AWS Account with Admin privileges

## Setup Raspberry Pi

* [Enable the camera](https://www.raspberrypi.org/documentation/configuration/camera.md)

## Install

### Clone repo

On the raspberry pi, `git clone` this repo.

```
$ git clone git@github.com:tleyden/eyepi.git
$ cd eyepi
```

### Install docker engine

Install docker engine using [the convenience script](https://docs.docker.com/engine/install/debian/#install-using-the-convenience-script).

Add the pi and greengrass users to the docker group:

```
$ sudo usermod -aG docker pi
$ sudo usermod -aG docker gg_user
```

### Build Docker image

```
$ docker build . -t eyepi
```

Make sure you can run the docker image:

```
$ docker run -it --device=/dev/video0:/dev/video0 eyepi
```

At this point you should see an error: `botocore.exceptions.NoCredentialsError: Unable to locate credentials` since you aren't passing in the AWS creds.

### Setup AWS Cloud

#### S3 Bucket

#### SNS topic

#### Lambda function

## References

* https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
* https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html
