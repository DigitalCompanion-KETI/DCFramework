# How to send handle variety data format

해당 가이드에서는 다양한 데이터 형태(비디오, 음성, 텍스트, 이미지)파일을 DCF에 어떻게 전달하고, 어떻게 사용하는지에 대해서 설명합니다.



해당 가이드에서 제공하는 입력데이터의 포맷은 다음과 같습니다.



1. 텍스트
2. 이미지
3. 음성파일(*.wav, etc)
4. 동영상파일(*.avi, *.mp4, etc)



## 1. 텍스트

 텍스트 데이터를 입력으로 하는 예제는 [1. Hello DCF](helloDCF.md)예제를 참고하시면 됩니다.

기본적인 함수 호출 방법은 다음과 같습니다.



```bash
$ dcf function call hello-dcf 
$ echo "Hello, DCF!" | dcf function call hello-dcf 

Hello, DCF
```



## 2. 이미지

이미지 데이터를 입력으로 하는 예제는 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)예제를 참고하시면 됩니다.

기본적인 로직은 다음과 같습니다

1. 이미지를 base64 문자열로 인코딩
2. `handler.py`에서 받은 base64 문자열을 이미지데이터로 디코딩
3. (.... 함수 코드)
4. 리턴



## 3. 음성파일 (*.wav, etc...)

음성 데이터를 입력으로 하는 예제는 다음과 같은 논리 흐름을 따릅니다.



1. *,wav파일을 전송
2. 함수 코드 실행
3. 리턴



음성파일을 다루는 방법을 `send-wav`튜토리얼로 설명드리도록 하겠습니다.

`send-wav` 함수는 다음과 같은 방법으로 음성파일이 잘 갔는지 확인합니다.



- Local device에서 음성데이터의 STFT 결과를 확인
- STFT Image를 반환하는 함수를 만든 후, 결과를 확인



`send-wav` 함수는 다음과 같은 논리 흐름을 따릅니다.



1. 음성파일을 수신
2. 수신된 음성파일을 STFT분석을 통해 이미지로 변환
3. 이미지를 base64 문자열로 인코딩
4. base64로 인코딩된 문자열을 리턴



### requirements.txt

`requirements.txt`파일은 다음과 같이 작성합니다.



```bash
numpy
matplotlib
scipy
pillow
```



### handler.py

함수 스크립트는 다음과 같습니다.



```python
# -*- coding: utf-8 -*-

import io
import os
import base64
import numpy as np
import matplotlib as m
from PIL import Image

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    m.use('Agg')

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(io.BytesIO(audiopath))

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    fig = plt.figure(figsize=(15, 7.5))
    plt.axis("off")
    plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    image = np.fromstring(fig.canvas.tostring_rgb() , dtype=np.uint8)
    image = image.reshape((h, w, 3))
    fig.clear()
    m.pyplot.close()

    return image


def Handler(req):
    rawwav = req.input
    image = plotstft(rawwav)
    
    im = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    base64Str = base64.b64encode(rawBytes.read()).decode("utf-8")


    return base64Str
```



### test.py 

로컬 디바이스에서 음성데이터를 확인할 수 있는 파이썬 스크립트 함수입니다.



```python
import io
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import base64
from PIL import Image

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    fig = plt.figure(figsize=(15, 7.5))
    plt.axis("off")
    plt.subplots_adjust(top = 1, bottom = 0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    image = np.fromstring(fig.canvas.tostring_rgb() , dtype=np.uint8)
    image = image.reshape((h, w, 3))
    fig.clear()
    plt.close()

    return image

image = plotstft("57007r.wav")

plt.figure()
plt.imshow(image)
plt.show()

im = Image.fromarray(image.astype("uint8"))
rawBytes = io.BytesIO()
im.save(rawBytes, "PNG")
rawBytes.seek(0)  # return to the start of the file
print(base64.b64encode(rawBytes.read()).decode("utf-8"))

```



### config.yaml

`send-wav`함수의 `config.yaml` 구성입니다.



```bash
  send-wav:
    runtime: python3
    desc: ""
    maintainer: ""
    handler:
      dir: ./send-wav
      file: handler.py
      name: Handler
    image: keti.asuscomm.com:5001/send-wav
    build_packages:
    - libfreetype6-dev
    - libpng12-dev
```



## 동영상 파일 (*.avi, *.mp4, etc)

동영상 파일을 전송해 분석하는 `ssd-video`함수를 생성하는 예제입니다.

`ssd-video`함수는 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md) 예제 기반으로 만들어져있습니다.



`ssd-video`함수는 다음과 같은 논리흐름을 따릅니다.



1. 비디오 파일을 전송
2. 비디오 파일로부터 매 프레임을 받아서 SSD모델을 이용해 프레임에 대한 추론
3. 추론된 결과를 json 파일로 통합
4. 통합된 json 결과를 리턴



### Dockerfile

```dockerfile
# Argruments from FROM
ARG PYTHON_VERSION=3.4

FROM python:${PYTHON_VERSION}

ARG ADDITIONAL_PACKAGE
RUN apt-get update && apt-get install -y  \
	build-essential \
	wget \
	tar \
        cmake \
        git \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        python3-dev \
        python3-numpy \
        python-numpy \
	${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION="3.4.2"

RUN pip3 install numpy
RUN pip install numpy

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.4 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3.4) \
  -DPYTHON_INCLUDE_DIR=$(python3.4 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.4 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make install \
&& rm /${OPENCV_VERSION}.zip \
&& rm -r /opencv-${OPENCV_VERSION}

ARG WATCHER_VERSION=0.1.0
ARG handler_file=handler.py
ARG handler_name=Handler

ENV HANDLER_DIR=/dcf/handler
ENV HANDLER_FILE=${HANDLER_DIR}/${handler_file}
ENV HANDLER_NAME=${handler_name}

# Get watcher
RUN mkdir -p ${HANDLER_DIR}
WORKDIR ${HANDLER_DIR}
RUN wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v${WATCHER_VERSION}/watcher${WATCHER_VERSION}-python.tar
RUN tar xvf watcher${WATCHER_VERSION}-python.tar
RUN pip install -r requirements.txt

# Copy handler
COPY . .
RUN touch ${HANDLER_DIR}/__init__.py
RUN pip install -r requirements.txt

HEALTHCHECK --interval=1s CMD [ -e /tmp/.lock ] || exit 1

ENTRYPOINT ["python"]
CMD ["server.py"]
```



### handler.py



```python
from __future__ import print_function
import io
import json
import numpy as np
import tensorflow as tf
import PIL

import cv2

from PIL import Image
from models.ssd.predict import predict

def Handler(req):

    videodata = req.input

    cap = cv2.VideoCapture(io.BytesIO(videodata))

    if not cap.isOpen():
        return "[cannot open video file]"

    result = dict()
    count = 0

    while True:
        ret, frame = video_capture.read()
    
        if ret:
            frame_result = predict(frame)
            count += 1
            result[str(count)] = frame_result
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        else:
            break

    video_capture.release()
  
    result = json.dumps(result)
    
    return result

```

