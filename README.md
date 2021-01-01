# eyepi

Security camera software that runs on Raspberry Pi with people detection

## Install

### How to Run TensorFlow Lite Object Detection Models on the Raspberry Pi

https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Raspberry_Pi_Guide.md

### Greengrass core v2

### Basic device setup pre-requisites

https://docs.aws.amazon.com/greengrass/latest/developerguide/setup-filter.rpi.html (and https://www.youtube.com/watch?v=LqOEfjygID0&t=440s)

NOTE: I installed java8 via `sudo apt install openjdk-8-jdk`, but the greengrass dependency checker can't find it.  Also, I think I already had java 11 installed because `java --version` returns `openjdk 11.0.9.1 2020-11-04`.  Fixed by running ` update-alternatives --config java` and selecting java 8.

### Greengrass core v2

Follow steps in:

https://docs.aws.amazon.com/greengrass/v2/developerguide/install-greengrass-core-v2.html

Command:

```
$ sudo -E java -Dlog.store=FILE -jar ./GreengrassCore/lib/Greengrass.jar --aws-region "us-west-2" --root "/greengrass/v2" --thing-name "EyePiGreengrassCore" --thing-group-name "EyePiGreengrassCoreGroup" --tes-role-name "EyePiGreengrassV2TokenExchangeRole" --tes-role-alias-name "EyePiGreengrassCoreTokenExchangeRoleAlias" --component-default-user "ggc_user:ggc_group" --provision true --setup-system-service true --deploy-dev-tools true
Added ggc_user to ggc_group
Provisioning AWS IoT resources for the device with IoT Thing Name: [EyePiGreengrassCore]...
Creating new IoT policy "GreengrassV2IoTThingPolicy"
Creating keys and certificate...
Attaching policy to certificate...
Creating IoT Thing "EyePiGreengrassCore"...
Attaching certificate to IoT thing...
Successfully provisioned AWS IoT resources for the device with IoT Thing Name: [EyePiGreengrassCore]!
Adding IoT Thing [EyePiGreengrassCore] into Thing Group: [EyePiGreengrassCoreGroup]...
Successfully added Thing into Thing Group: [EyePiGreengrassCoreGroup]
Setting up resources for aws.greengrass.TokenExchangeService ...
TES role alias "EyePiGreengrassCoreTokenExchangeRoleAlias" does not exist, creating new alias...
TES role "EyePiGreengrassV2TokenExchangeRole" does not exist, creating role...
IoT role policy "GreengrassTESCertificatePolicyEyePiGreengrassCoreTokenExchangeRoleAlias" for TES Role alias not exist, creating policy...
Attaching TES role policy to IoT thing...
IAM role policy for TES "EyePiGreengrassV2TokenExchangeRoleAccess" created. This policy DOES NOT have S3 access, please modify it with your private components' artifact buckets/objects as needed when you create and deploy private components
Attaching IAM role policy for TES to IAM role for TES...
Configuring Nucleus with provisioned resource details...
Downloading Root CA from "https://www.amazontrust.com/repository/AmazonRootCA1.pem"
Created device configuration
Successfully configured Nucleus with provisioned resource details!
Creating a deployment for Greengrass first party components to the thing group
Configured Nucleus to deploy aws.greengrass.Cli component
Successfully set up Nucleus as a system service
```

Follow steps in:

https://docs.aws.amazon.com/greengrass/v2/developerguide/getting-started.html

## Install docker engine

https://docs.docker.com/engine/install/debian/  (based on https://docs.aws.amazon.com/greengrass/v2/developerguide/run-docker-container.html)

## Get hello-world docker component working

Add ggc_user to docker group:

```
$ usermod -aG docker ggc_user
```

This step isn't in the instructions, but I think it might fix this error:

```
2021-01-01T16:00:30.106Z [WARN] (Copier) com.example.MyDockerComponent: stderr. Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post http://%2Fvar%2Frun%2Fdocker.sock/v1.24/images/load?quiet=1: dial unix /var/run/docker.sock: connect: permission denied. {scriptName=services.com.example.MyDockerComponent.lifecycle.Install.Script, serviceName=com.example.MyDockerComponent, currentState=NEW}
```



```
mkdir -p artifacts/com.example.MyDockerComponent/1.0.0
docker save hello-world > artifacts/com.example.MyDockerComponent/1.0.0/hello-world.tar
```

```
vi recipes/com.example.MyDockerComponent-1.0.0.yaml
```

and paste in:

```
---
RecipeFormatVersion: '2020-01-25'
ComponentName: com.example.MyDockerComponent
ComponentVersion: '1.0.0'
ComponentDescription: A component that runs a Docker container.
ComponentPublisher: Amazon
Manifests:
  - Platform:
      os: linux
    Lifecycle:
      Install:
        Script: docker load -i {artifacts:path}/hello-world.tar
      Run:
        Script: docker run --rm hello-world
```

Wait a few mins, then `sudo cat /greengrass/v2/logs/com.example.MyDockerComponent.log`


## Publish iot core message from docker

1. Follow steps to setup docker for IPC
1. Create a docker container that has python3 installed, then `python3 -m pip install awsiotsdk` based on https://github.com/aws/aws-iot-device-sdk-python-v2
1. Publish and iot core message based on https://docs.aws.amazon.com/greengrass/v2/developerguide/interprocess-communication.html#ipc-iot-core-mqtt

## References


