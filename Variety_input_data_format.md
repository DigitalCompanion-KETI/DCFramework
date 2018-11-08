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

이미지 데이터를 입력으로 하는 예제는 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)예제를
참고하시면 됩니다. 기본적인 로직은 다음과 같습니다

1. 이미지를 base64 문자열로 인코딩
2. `handler.py`에서 받은 base64 문자열을 이미지데이터로 디코딩
3. (.... 함수 코드)
4. 리턴



함수 호출은 다음과 같은 방법으로 호출합니다.



```bash
cat 000001.jpg | base64 | dcf function call ssd-test

>>> 
[{"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}, {"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}]
```

​    

## 3. 음성파일 (*.wav, etc...)

음성파일을 다루는 방법은 `send-wav`함수를 생성하는 튜토리얼로 설명드리도록 하겠습니다.
`send-wav` 함수는 다음과 같은 방법으로 음성파일을 전달하고, 해당 음성파일이 잘 갔는지 리턴값을 보고 확인합니다.
음성파일은 [다음 링크](https://www.loc.gov/item/00694083/)에서 다운로드 받을 수 있습니다.



- Local device에서 음성데이터의 STFT 결과를 확인
- STFT Image를 반환하는 함수를 만든 후, 결과를 확인
- Local device와 함수 호출 결과의 이미지 결과값이 같은지 확인


> Note
>
> `STFT(Short Time Fourier Transform)`란 시간의 경과에 따라 변하는 로컬 영역 신호의 정현파 주파수 및 위상 내용을
> 결정하는데 사용되는 푸리에 변환입니다. 


`send-wav` 함수는 다음과 같은 논리 흐름을 따릅니다.



1. 음성파일을 수신
2. 수신된 음성파일을 STFT분석을 통해 이미지로 변환
3. 이미지를 base64 문자열로 인코딩
4. base64로 인코딩된 문자열을 리턴



### requirements.txt

DCF에 음성파일을 전달하고 문자열을 리턴받기 까지의 과정을 수행하기 위해 
우선 `requirements.txt`파일에 다음과 같이 설치가 요구되는 패키지를 작성합니다. 



```bash
numpy
matplotlib
scipy
pillow
```



### handler.py

음성파일을 통해 문자열을 리턴하기 까지의 동작이 실행되는 함수 스크립트는 다음과 같습니다.
해당 함수 스크립트는 [해당 링크](http://www.frank-zalkow.de/en/code-snippets/create-audio-spectrograms-with-python.html?i=1)를 
참고하였습니다.



```python
# -*- coding: utf-8 -*-

import io
import os
import base64
import numpy as np
import matplotlib as m
from PIL import Image

# 함수에서는 디스플레이가 불가능하니, 디스플레이 옵션을 사용하지 않게 변경해줍니다.
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
    # 받은 audio raw bytes파일을 wav.read함수를 이용하여 load
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
    # 입력값을 받아, 그대로 plotstft 함수로 전달
    rawwav = req.input
    image = plotstft(rawwav)
    
    # numpy image를 PIL.Image객체로 변환
    im = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    
    # PIL Image를 PNG 포맷으로 변환
    im.save(rawBytes, "PNG")
    rawBytes.seek(0)  # return to the start of the file
    
    # PNG포맷으로 변환된 이미지를 base64 문자열로 인코딩
    base64Str = base64.b64encode(rawBytes.read()).decode("utf-8")


    return base64Str
```

> Note

```python
import matplotlib as m

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    m.use('Agg')
```

> 위와 같이 `matplotlib` 패키지를 import 하지 않고 조건문을 작성하지 않으면 다음과 같은 에러가 발생합니다. 
> 이는 함수에서는 디스플레이가 불가능하기 때문에 발생하는 에러이기 때문에, 
> 디스플레이 옵션을 사용하지 않게 변경을 해야 음성파일을 통한 문자열을 리턴받을 수 있습니다.

```bash
>> Error

_tkinter.TclError: couldn't connect to display "localhost:10.0"
```

### test.py 

로컬 디바이스에서 음성데이터를 확인할 수 있는 파이썬 스크립트 함수입니다.
이는 `handler.py` 스크립트의 결과를 로컬 디바이스에서 확인해보기 위함입니다. 
다음의 스크립트를 원하는 디렉토리에 생성한 후, 로컬 디바이스에서 실행합니다. 



```python
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

# 음성파일의 위치를 입력값으로 plotstft 함수 호출
image = plotstft("57007r.wav")

# STFT 결과 이미지 확인
plt.figure()
plt.imshow(image)
plt.show()

# 해당 이미지를 base64 문자열로 인코딩 후, 결과 확인
im = Image.fromarray(image.astype("uint8"))
rawBytes = io.BytesIO()
im.save(rawBytes, "PNG")
rawBytes.seek(0)  # return to the start of the file
print(base64.b64encode(rawBytes.read()).decode("utf-8"))

```


### test.py 결과 확인

다음과 같은 명령어로 test.py파일을 실행해볼 수 있습니다.



```bash
python3 test.py
```



**결과**

![result.jpg](https://user-images.githubusercontent.com/13328380/48113924-42503e80-e2a0-11e8-9681-6cd87bd61f4f.jpg)



### config.yaml

`send-wav`함수의 `config.yaml` 구성입니다.



`libfreetype6-dev`, `libpng-dev` package가 필요하므로, `build_packages`에 명시해줍니다.

```bash
functions
  send-wav:
    runtime: python
    desc: ""
    maintainer: ""
    handler:
      dir: ./send-wav
      file: handler.py
      name: Handler
    image: keti.asuscomm.com:5001/send-wav
    build_args:
    - PYTHON_VERSION=3.4
    build_packages:
    - libfreetype6-dev
    - libpng-dev
```



### 함수 호출

다음과 같이 셋팅을 완료하고 함수를 배포했다면, 다음과 같이 음성파일을 전달하여 결과값을 확인할 수 있습니다.



```bash
$ cat 57007r.wav | dcf function call send-wav

>>
iVBORw......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
```



### 결과 확인

결과값으로 온 문자열 값을 모두 복사해서 [다음 사이트](https://codebeautify.org/base64-to-image-converter)에서 
base64 문자열을 이미지로 복원해서 확인할 수 있습니다.



**결과**

![function result.jpg](https://user-images.githubusercontent.com/13328380/48114112-171a1f00-e2a1-11e8-9347-fd31f7324577.png)_





## 동영상 파일 (*.avi, *.mp4, etc)

동영상 파일을 전송해 분석하는 `ssd-video`함수를 생성하는 예제입니다.
`ssd-video`함수는 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md) 예제 기반으로 만들어져있습니다.
따라서 해당 예제에서는 SSD 모델을 설정하는 방법은 동일하며, 변경된 `Dockerfile`과 `Handler.py`에 대해서만 설명하겠습니다.

`ssd-video` 폴더의 계층은 다음과 같습니다. 
models 디렉토리 설정은 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md) 예제와 동일합니다. 

```bash
ssd-video
├── Dockerfile
├── handler.py
├── models
│   └── ssd
└── requirements.txt
```


`ssd-video`함수는 다음과 같은 논리흐름을 따릅니다.

1. 비디오 파일을 전송
2. 비디오 파일로부터 10번째 프레임까지만 받아서 SSD모델을 이용해 프레임에 대한 추론
3. 추론된 결과를 json 파일로 통합
4. 통합된 json 결과를 리턴



동영상 파일은 [다음 링크](https://videos.pexels.com/videos/view-of-the-lake-and-mountains-from-a-park-1466209)에서 다운받을 수 있습니다.



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
        libgtk2.0-dev \
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
import os
import sys
sys.path.append("./models/ssd")
import io
import json
import numpy as np
import tensorflow as tf
import PIL


import cv2

from PIL import Image
from models.ssd.predict import predict

def Handler(req):

    videodata = io.BytesIO(req.input).read()

    if os.path.isfile("video.mp4"):
        os.remove("video.mp4")

    with open("video.mp4", "wb") as f:
        f.write(videodata)
    

    cap = cv2.VideoCapture("video.mp4")

    if not cap.isOpened():
        return "[cannot open video file]"

    result = dict()
    count = 0

    while True:
        ret, frame = cap.read()
    
        if ret:
            if count >= 10:
                break

            frame_result = predict(frame)
            count += 1
            result[str(count)] = frame_result

        else:
            break

    cap.release()
    result = json.dumps(result)

    if os.path.isfile("video.mp4"):
        os.remove("video.mp4")
    
    return result
```



### 함수 호출



`ssd-video`함수에 대해서 셋팅 및 배포를 완료하였다면, 다음과 같은 명령어로 함수를 호출할 수 있습니다.



```bash
$ cat Pexels\ Videos\ 1466209.mp4 | dcf function call ssd-video

>>
{"8": [{"confidence": "0.6046466", "xmax": "718", "ymax": "379", "ymin": "379", "xmin": "718", "class": "15"}, {"confidence": "0.6046466", "xmax": "718", "ymax": "379", "ymin": "379", "xmin": "718", "class": "15"}, {"confidence": "0.6046466", "xmax": "718", "ymax": "379", "ymin": "379", "xmin": "718", "class": "15"}, {"confidence": "0.6046466", "xmax": "718", "ymax": "379", "ymin": "379", "xmin": "718", "class": "15"}], "4": [{"confidence": "0.6336566", "xmax": "722", "ymax": "383", "ymin": "383", "xmin": "722", "class": "15"}, {"confidence": "0.6336566", "xmax": "722", "ymax": "383", "ymin": "383", "xmin": "722", "class": "15"}, {"confidence": "0.6336566", "xmax": "722", "ymax": "383", "ymin": "383", "xmin": "722", "class": "15"}], "10": [{"confidence": "0.7154824", "xmax": "527", "ymax": "368", "ymin": "368", "xmin": "527", "class": "15"}, {"confidence": "0.7154824", "xmax": "527", "ymax": "368", "ymin": "368", "xmin": "527", "class": "15"}, {"confidence": "0.7154824", "xmax": "527", "ymax": "368", "ymin": "368", "xmin": "527", "class": "15"}], "1": [{"confidence": "0.88071", "xmax": "726", "ymax": "385", "ymin": "385", "xmin": "726", "class": "15"}, {"confidence": "0.88071", "xmax": "726", "ymax": "385", "ymin": "385", "xmin": "726", "class": "15"}, {"confidence": "0.88071", "xmax": "726", "ymax": "385", "ymin": "385", "xmin": "726", "class": "15"}], "7": [{"confidence": "0.6155173", "xmax": "513", "ymax": "335", "ymin": "335", "xmin": "513", "class": "15"}, {"confidence": "0.6155173", "xmax": "513", "ymax": "335", "ymin": "335", "xmin": "513", "class": "15"}, {"confidence": "0.6155173", "xmax": "513", "ymax": "335", "ymin": "335", "xmin": "513", "class": "15"}], "9": [{"confidence": "0.5337455", "xmax": "707", "ymax": "361", "ymin": "361", "xmin": "707", "class": "15"}, {"confidence": "0.5337455", "xmax": "707", "ymax": "361", "ymin": "361", "xmin": "707", "class": "15"}, {"confidence": "0.5337455", "xmax": "707", "ymax": "361", "ymin": "361", "xmin": "707", "class": "15"}, {"confidence": "0.5337455", "xmax": "707", "ymax": "361", "ymin": "361", "xmin": "707", "class": "15"}], "2": [{"confidence": "0.8497871", "xmax": "726", "ymax": "384", "ymin": "384", "xmin": "726", "class": "15"}, {"confidence": "0.8497871", "xmax": "726", "ymax": "384", "ymin": "384", "xmin": "726", "class": "15"}, {"confidence": "0.8497871", "xmax": "726", "ymax": "384", "ymin": "384", "xmin": "726", "class": "15"}], "6": [{"confidence": "0.60376704", "xmax": "719", "ymax": "392", "ymin": "392", "xmin": "719", "class": "15"}, {"confidence": "0.60376704", "xmax": "719", "ymax": "392", "ymin": "392", "xmin": "719", "class": "15"}, {"confidence": "0.60376704", "xmax": "719", "ymax": "392", "ymin": "392", "xmin": "719", "class": "15"}], "3": [{"confidence": "0.73087525", "xmax": "524", "ymax": "519", "ymin": "519", "xmin": "524", "class": "2"}, {"confidence": "0.73087525", "xmax": "524", "ymax": "519", "ymin": "519", "xmin": "524", "class": "2"}, {"confidence": "0.73087525", "xmax": "524", "ymax": "519", "ymin": "519", "xmin": "524", "class": "2"}], "5": [{"confidence": "0.75017697", "xmax": "723", "ymax": "388", "ymin": "388", "xmin": "723", "class": "15"}, {"confidence": "0.75017697", "xmax": "723", "ymax": "388", "ymin": "388", "xmin": "723", "class": "15"}, {"confidence": "0.75017697", "xmax": "723", "ymax": "388", "ymin": "388", "xmin": "723", "class": "15"}]}
```

