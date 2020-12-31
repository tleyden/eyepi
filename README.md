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

https://docs.aws.amazon.com/greengrass/v2/developerguide/install-greengrass-core-v2.html

Command:

```
$ sudo -E java -Dlog.store=FILE -jar ./GreengrassCore/lib/Greengrass.jar --aws-region "us-west-2" --root "/greengrass/v2" --thing-name "EyePiGreengrassCore" --thing-group-name "EyePiGreengrassCoreGroup" --tes-role-name "EyePiGreengrassV2TokenExchangeRole" --tes-role-alias-name "EyePiGreengrassCoreTokenExchangeRoleAlias" --component-default-user "ggc_user:ggc_group" --provision true --setup-system-service true --deploy-dev-tools true
```

## References


