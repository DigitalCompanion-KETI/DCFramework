# More Example in DCF

본 예제는 디지털 동반자 프레임워크를 이용하여 여러 응용 어플리케이션을 작성하는 방법에 대해서 소개한다. 예제는 사용자의 개발환경에서 함수를 테스트하는 단계까지만 예제의 구성은 아래와 같다.



- Hello DCF (Text)
- Object Detection using SSD; Single Shot Multibox-Detector with GPU(Image)
- STFT; Short Time Fourier Transform (Audio)
- Streaming
  - Text
  - Video
  - Audio

<br/>

## Hello DCF

"Hello DCF"는 함수를 호출할 때, 반환값을 "Hello DCF"로 돌려주는 예제이다.



### Init

hello-dcf라는 함수를 생성한다

```bash
$ dcf-cli function init hello-dcf --runtime python
>>
Directory: hello-dcf is created.
Function handler created in directory: hello-dcf/src
Rewrite the function handler code in hello-dcf/src directory
Config file written: config.yaml
```



### Write function

`src/handler.py`에서 반환값을 "Hello DCF"라고 변경한다

```bash
$ cd hello-dcf/src
$ vim handler.py
```

```python
class Handler:
    def __init__(self):
        pass

    def __call__(self, req):
        # change this line
        return "Hello DCF"
```



### Build function

작성한 함수를 빌드한다

```bash
$ cd hello-dcf
$ dcf-cli function build -f config.yaml -v
>>
Building function (hello-dcf) ...
Sending build context to Docker daemon  8.192kB
Step 1/45 : ARG ADDITIONAL_PACKAGE
Step 2/45 : ARG REGISTRY
Step 3/45 : ARG PYTHON_VERSION
...
```



### Test function

빌드한 함수를 사용자 디바이스에서 테스트한다

```bash
$ echo "Hello" | dcf-cli function run hello-dcf
>>
Checking VGA Card...
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GM206GL [Quadro M2000] [10de:1430] (rev a1)
01:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:0fba] (rev a1)


Checking Driver Info...

==============NVSMI LOG==============

Timestamp                           : Mon Oct  7 19:59:26 2019
...
Running image (keti.asuscomm.com:5001/hello-dcf) in local
Starting [dcf-watcher] server ...
Call hello-dcf in user's local
Handler request: Hello

Handler reply: Hello DCF
```





<br/>

## Object Detection using SSD; Single Shot Multibox Detector with GPU

인공지능 모델 중 객체 검출 모델인 SSD; Single Shot Multibox Detector를 이용하여 객체의 박스정보를 반환하는 예제이다. 이번 예제에서는 GPU를 사용하는 방법을 함께 소개한다. 따라서 이 예제를 실행하기 위해서는 컴퓨터에 GPU 장착 및 그래픽 드라이버, 엔비디아 도커가 설치되어있어야한다.



![concept](https://user-images.githubusercontent.com/13328380/47901830-507b1500-dec4-11e8-9a3d-afba90719b9d.png)



위의 그림은 이번 예제가 동작하는 방식에 대해서 잘 표현하고있다. 이 예제는 아래와 같은 순서로 구동된다.



- Client
  1. Image를 base64로 인코딩
  2. 인코딩된 이미지를 DCF로 전송

- DCF
  1. base64로 받은 Image 데이터를 디코딩
  2. 디코딩 된 데이터를 Image로 변환
  3. 인공지능 모델에 Image 정보를 입력으로 사용하여 객체 정보를 획득
  4. 획득한 객체 정보를 JSON으로 변환하여 Client 전송



### Init

ssd-gpu라는 함수를 생성한다

```bash
$ dcf-cli function init ssd-gpu
>>
Directory: ssd-gpu is created.
Function handler created in directory: ssd-gpu
Rewrite the function handler code in ssd-gpu/src directory
Config file written: config.yaml
```



### Write function

SSD 모델이 위지할 디렉토리를 만든다.

```bash
$ cd ssd-gpu/src
$ mkdir models
```



#### Clone SSD TensorFlow implementation

SSD의 TensorFlow 구현체를 받는다.

```bash
$ cd models
$ git clone https://github.com/balancap/SSD-Tensorflow.git ssd
```



#### Unzip checkpoint file

checkpoint 폴더 안에 있는 모델의 가중치 파일을 압축해제한다.

```bash
$ cd ssd/checkpoint
$ unzip ssd_300_vgg.ckpt.zip
>>>
Archive:  ssd_300_vgg.ckpt.zip
  inflating: ssd_300_vgg.ckpt.data-00000-of-00001  
  inflating: ssd_300_vgg.ckpt.index 
```



#### Modify Dockerfile for install OpenCv

Dockerfile의 `2-Stage` 이후의 아래 코드를 다음과 같이 변경한다

```dockerfile
...

# 2-Stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Arguments for Nvidia-Docker
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG CUDA_VERSION_BACKUP

RUN echo "/usr/local/cuda-${CUDA_VERSION_BACKUP}/extras/CUPTI/lib64" > /etc/ld.so.conf.d/cupti.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	wget \
	tar \
	libgomp1 \
        python-setuptools \
        libgtk2.0-dev \
        libcudnn7=${CUDNN_VERSION}-1+cuda${CUDA_VERSION_BACKUP} \
        python${PYTHON_VERSION} \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-setuptools \
        python3-tk \
        cmake \
        unzip \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libgtk2.0-dev \
	${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

ENV OPENCV_VERSION="3.4.2"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && \
    cd opencv-${OPENCV_VERSION} && \
    mkdir build && \
    cd build && \
cmake -DBUILD_TIFF=ON \
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
  -DCMAKE_INSTALL_PREFIX=$(python${PYTHON_VERSION} -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python${PYTHON_VERSION}) \
  -DPYTHON_INCLUDE_DIR=$(python${PYTHON_VERSION} -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python${PYTHON_VERSION} -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. &&\
make install && \
cd ../.. && \
rm ${OPENCV_VERSION}.zip && \
rm -r opencv-${OPENCV_VERSION}

...
```



#### Configure GPU

GPU 사용을 허용하기 위해 `ssd-gpu/config.yaml`파일에서 `limits`에 `gpu`의 값을 1로 변경한다.

```yaml
functions:
  ssd-gpu:
    runtime: python
    desc: ""
    maintainer: ""
    handler:
      dir: ./src
      file: ""
      name: Handler
    docker_registry: keti.asuscomm.com:5001
    image: keti.asuscomm.com:5001/ssd-gpu
    limits:
      memory: ""
      cpu: ""
      gpu: "1"
    build_args:
    - CUDA_VERSION=9.0
    - CUDNN_VERSION=7.4.1.5
    - UBUNTU_VERSION=16.04
dcf:
  gateway: keti.asuscomm.com:32222
```



#### Write Prediction.py

함수의 `Handler`클래스에서 쉽게 사용할 수 있게 `predict.py`파일을 `ssd-gpu/src/models/ssd`안에 작성한다.

인공지능 모델의 경우 모델 생성 및 가중치 로드를 1회만 할 수 있도록 생성자와 호출부를 잘 분리해야한다.



```bash
$ cd ssd-gpu/src/models/ssd
$ vim prediction.py
```

```python
import sys
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

class Predictor:
    def __init__(self):
        self_dir = os.path.dirname(os.path.realpath(__file__))

        slim = tf.contrib.slim
    
        # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        # Input placeholder.
        net_shape = (300, 300)
        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(image_pre, 0)

        # Define the SSD model.
        #reuse = True if 'ssd_net' in locals() else None
        reuse = False
        self.ssd_net = ssd_vgg_300.SSDNet()

        with slim.arg_scope(self.ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = self.ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        ckpt_filename = os.path.join(self_dir, 'checkpoints/ssd_300_vgg.ckpt')
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filename)

        # SSD default anchor boxes.
        self.ssd_anchors = self.ssd_net.anchors(net_shape)

    def __call__(self, image):

        # Main image processing routine.
        def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
            # Run SSD network.
            rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                       feed_dict={self.img_input: img})

            # Get classes and bboxes from the net outputs.
            rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

            rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
            rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
            rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
            # Resize bboxes to original image shape. Note: useless for Resize.WARP!
            rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
            return rclasses, rscores, rbboxes

        None_flag = False
        # Test on some demo image and visualize output.
        if image is None:
            None_flag = True
            path = 'models/ssd/demo/'
            image_names = sorted(os.listdir(path))
            image = mpimg.imread(path + image_names[-5])


        rclasses, rscores, rbboxes =  process_image(image)

        if None_flag:
            visualization.bboxes_draw_on_img(image, rclasses, rscores, rbboxes, visualization.colors_plasma)
            visualization.plt_bboxes(image, rclasses, rscores, rbboxes)

        height = image.shape[0]
        width = image.shape[1]
        colors = dict()

        obj = list()
        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                score = rscores[i]
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
            
                obj_info = dict()
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)

                class_name = str(cls_id)

                obj_info["class"] = class_name
                obj_info["confidence"] = str(score)
                obj_info["xmin"] = str(xmin)
                obj_info["ymin"] = str(ymin)
                obj_info["xmax"] = str(xmax)
                obj_info["ymax"] = str(ymax)
                obj.append(obj_info)
    
        return obj

if __name__ == "__main__":
    predictor = Predictor()
    print(predictor(image=None))
```



#### Modify Handler.py

이제 SSD 모델을 함수에서 추론할 수 있게 `ssd-gpu/src/handler.py`을 수정한다

```python
import os
import sys
import io
import json
import base64
import numpy as np
import PIL
import tensorflow as tf
from PIL import Image
import imp

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/ssd"))
from predict import Predictor

class Handler:
    def __init__(self):
        self.predictor = Predictor()        

    def __call__(self, req):

        base64Str = req.input
        imageData = base64.decodestring(base64Str)
        img = Image.open(io.BytesIO(imageData))
        width, height = img.size
    
        img = np.array(img)

        result = self.predictor(img)
        result = json.dumps(result)  
    
        return result
```



#### requirements.txt

함수가 배포될 때, 필요한 패키지를 설치할 수 있도록 `ssd-gpu/requirements.txt`파일을 아래와 같이 수정한다

```
matplotlib
pillow
scipy
numpy
tensorflow-gpu==1.11.0
```



### Build function

작성한 함수를 빌드한다

```bash
$ dcf-cli function build -f config.yaml -v
>>>
Building function (ssd-gpu) ...
Sending build context to Docker daemon  325.8MB
Step 1/47 : ARG ADDITIONAL_PACKAGE
Step 2/47 : ARG REGISTRY
...
```



### Test Function

빌드한 함수를 테스트한다

```bash
$ cat ssd-gpu/src/models/ssd/demo/000001.jpg | base64 | dcf-cli function run ssd-gpu
>>>
[{"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}, {"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}]
```





<br/>

## STFT; Short Time Fourier Transform

STFT 예제는  `*.wav`파일을 전송하고 결과값으로 STFT; Short Time Fourier Transform을 거친 영상을 결과값으로 받는 예제이다. 본 예제에서 사용할 음성 파일은 [링크](https://www.loc.gov/item/00694083/)에서 다운로드 할 수 있다.



함수의 구동 순서는 아래와 같다.

- Client
  - `*.wav`파일을 전송
- DCF
  - `*.wav`파일 수신
  - STFT 알고리즘 통과
  - STFT 결과 이미지를 반환



### Init

stft라는 함수를 생성한다

```bash
$ dcf-cli function init stft
>>>
Directory: stft is created.
Function handler created in directory: stft
Rewrite the function handler code in stft/src directory
Config file written: config.yaml
```



### Write function

#### Dockerfile

필요한 패키지 설치를 위해서 Dockerfile에서 apt-get으로 `libfreetype6-dev`, `libpng-dev` 패키지를 설치한다

```dockerfile
# 2-Stage
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Arguments for Nvidia-Docker
ARG CUDA_VERSION
ARG CUDNN_VERSION
ARG CUDA_VERSION_BACKUP

RUN echo "/usr/local/cuda-${CUDA_VERSION_BACKUP}/extras/CUPTI/lib64" > /etc/ld.so.conf.d/cupti.conf

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	wget \
	tar \
	libgomp1 \
        python-setuptools \
        libgtk2.0-dev \
        libcudnn7=${CUDNN_VERSION}-1+cuda${CUDA_VERSION_BACKUP} \
        python${PYTHON_VERSION} \
        python3-dev \
        python3-numpy \
        python3-pip \
        python3-tk \
        cmake \
        unzip \
        pkg-config \
        libfreetype6-dev \
        libpng-dev \
	${ADDITIONAL_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*
```



#### handler.py

아래와 같이 `handler.py` 파일을 작성한다

```python
# -*- coding: utf-8 -*-

import io
import os
import base64
import numpy as np
import matplotlib as m
from PIL import Image

# 함수에서는 디스플레이가 불가능하니, 디스플레이 옵션을 사용하지 않게 변경해준다
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

class Handler:
    def __init__(self):
        pass

    def __call__(self, req):
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



#### requirements.txt

필요한 패키지 파일을 requirements.txt에 명시한다

```bash
numpy
matplotlib
scipy
pillow
```



### Build function

작성한 함수를 빌드한다

```bash
$ dcf-cli function builf -f config.yaml -v
>>>
Building function (stft) ...
Sending build context to Docker daemon  325.8MB
Step 1/47 : ARG ADDITIONAL_PACKAGE
Step 2/47 : ARG REGISTRY
...
```



### Test Function

빌드한 함수를 테스트한다. 

```bash
$ cat 57007r.wav | dcf function call send-wav
>>
iVBORw......XqWY4HTbaj+3/xNf634UHr7xLvnzb2f/zL8Cdf6j14yamriQAAAABJRU5ErkJggg==
```



출력된 결과를 Base64 디코딩하여 이미지를 확인해보면 아래와 같은 이미지를 확인할 수 있다. Base64 디코딩은 [여기](https://codebeautify.org/base64-to-image-converter)에서 할 수 있다.

![stft](https://user-images.githubusercontent.com/13328380/48114112-171a1f00-e2a1-11e8-9347-fd31f7324577.png)