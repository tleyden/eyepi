# eyepi

Security camera software that runs on Raspberry Pi with people detection

When a person is detected by the model send a notification with a 5 second video clip stored on AWS S3.

Send at most one notification per 5 minute period.

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

Make sure you can run the docker image .. TODO

### Setup AWS Cloud

#### S3 Bucket

#### SNS topic

#### Lambda function

### (Optional) Install AWS Greengrass core v2

AWS Greengrass is an IoT framework that helps handle authentication issues between the device and the cloud.  It also handles Over-the-air updates.

#### Basic device setup pre-requisites for Greengrass

https://docs.aws.amazon.com/greengrass/latest/developerguide/setup-filter.rpi.html (and https://www.youtube.com/watch?v=LqOEfjygID0&t=440s)

NOTE: I installed java8 via `sudo apt install openjdk-8-jdk`, but the greengrass dependency checker can't find it.  Also, I think I already had java 11 installed because `java --version` returns `openjdk 11.0.9.1 2020-11-04`.  Fixed by running ` update-alternatives --config java` and selecting java 8.

#### Greengrass core v2

Follow steps in:

1. https://docs.aws.amazon.com/greengrass/v2/developerguide/install-greengrass-core-v2.html

1. https://docs.aws.amazon.com/greengrass/v2/developerguide/getting-started.html

## References

* https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
* https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html
