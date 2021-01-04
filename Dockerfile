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
# RUN pip3 install opencv-python==3.4.6.27

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
# Will change to just 'pip3 install tensorflow' once newer versions of TF are added to piwheels
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl
