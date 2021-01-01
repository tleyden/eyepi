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

## Publish iot core message

1. Figure out how to use a virtualenv from a greeengrass component using this hack: https://unix.stackexchange.com/questions/209646/how-to-activate-virtualenv-when-a-python-script-starts (or if that fails, use a docker container -- I will not pollute the default python3!!)
1. `python3 -m pip install awsiotsdk` based on https://github.com/aws/aws-iot-device-sdk-python-v2

## References


