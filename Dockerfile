FROM arm32v7/python:3.7.9-buster

RUN wget https://archive.raspbian.org/raspbian.public.key -O - | apt-key add -
RUN wget https://archive.raspberrypi.org/debian/raspberrypi.gpg.key -O - | apt-key add -
RUN echo "deb http://raspbian.raspberrypi.org/raspbian/ buster main contrib non-free rpi" >> /etc/apt/sources.list

RUN apt-get update && apt-get install -y emacs25-nox \
    libjpeg-dev \
    libtiff5-dev \
    libjasper-dev \
    libpng12-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    qt4-dev-tools \
    libatlas-base-dev \
    libgtk3.0

# This is needed to in order to pip3 install opencv-python==3.4.11.45
RUN mkdir -p /root/.config/pip/
COPY docker/pip.conf /root/.config/pip/

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install boto3

# Need to get an older version of OpenCV because version 4 has errors - see https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/get_pi_requirements.sh
RUN pip3 install opencv-python==3.4.11.45

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# TODO: change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

RUN mkdir /root/coco_ssd_mobilenet
RUN wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -O /root/coco_ssd_mobilenet/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
RUN unzip /root/coco_ssd_mobilenet/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -d /root/coco_ssd_mobilenet/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29

# Copy in eyepi.py script
COPY eyepi.py /root/eyepi.py

CMD ["/root/eyepi.py --modeldir /root/coco_ssd_mobilenet/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"]
ENTRYPOINT ["python3"]

# TODO: parameterize target s3 bucket name
