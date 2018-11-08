# SSD(Object Detection)  Component 

해당 가이드는 딥러닝 모델 중 Object Detection을 하는 SSD(Single Shot MultiBox Detector)모델을 DCF를 통해 배포하는 튜토리얼 가이드입니다.



Object Detection 콤포넌트는 아래 그림과 같이 `Image`를 입력으로 받아 이를 `class, confidence score, box coordinates`정보를 `JSON`포맷으로 리턴하는 예제입니다.



해당 컴포넌트의 논리 흐름은 다음과 같은 흐름을 따릅니다.

1. Image를 base64로 인코딩 (DCF Client)
2. base64로 받은 Image 데이터를 디코딩 (Component)
3. 디코딩 된 데이터를 Image로 변환 (Component)
4. 변환된 Image를 입력으로 Model 추론 (Component)
5. 추론 결과를 JSON으로 변환 및 리턴 (Component)



![Deploy Object Detection Model in DCF](https://user-images.githubusercontent.com/13328380/47901830-507b1500-dec4-11e8-9a3d-afba90719b9d.png)



튜토리얼 문서는 DCF를 이용하여 SSD model을 구현하는 방법을 순차적으로 기술해놓았으므로, 상단에서 하단으로 가이드 문서를 따라가면 해당 함수 컴포넌트를 생성하고 배포할 수 있습니다.

​    

## DCF 함수 컴포넌트 생성

먼저 DCF CLI를 이용하여 python 함수 컴포넌트를 생성합니다.



```bash
$ dcf function init --runtime python ssd-image

>>>
Folder: ssd-image created.
Function handler created in folder: ssd-image
Rewrite the function handler code in ssd-image folder
Config file written: config.yaml
```



생성 후에, 다음과 같은 폴더 구조를 확인할 수 있습니다.



```bash
ssd-image
├── Dockerfile
├── handler.py
└── requirements.txt
```

​    

## Installing SSD Tensorflow implementation

​    

### 1. SSD Tensorflow구현체 설치를 위한 폴더 생성

먼저 생성된 python함수 컴포넌트에 다음과 같이 폴더를 생성합니다.

`ssd-image -> models -> ssd`



```bash
$ cd ssd-image
$ mkdir models
$ cd models
$ mkdir ssd
```



이렇게 폴더를 만들면, `ssd-image`폴더 계층은 다음과 같아집니다.

```bash
ssd-image
├── Dockerfile
├── handler.py
├── models
│   └── ssd
└── requirements.txt
```

​    

### 2. Clone SSD Tensorflow Implementation from github

​    

#### SSD Tensorflow Implementation Clone

ssd 폴더로 진입하여, [SSD Tensorflow 구현체(Tensorflow 1.11.0)](https://github.com/balancap/SSD-Tensorflow)의 코드를 clone합니다.



```bash
$ cd models/ssd
$ git init
$ git remote add origin https://github.com/balancap/SSD-Tensorflow.git
$ git pull origin master

>> 
Initialized empty Git repository in ../ssd-image/models/ssd/.git/
remote: Enumerating objects: 809, done.
remote: Total 809 (delta 0), reused 0 (delta 0), pack-reused 809
Receiving objects: 100% (809/809), 113.09 MiB | 1.20 MiB/s, done.
Resolving deltas: 100% (547/547), done.
From https://github.com/balancap/SSD-Tensorflow
 * branch            master     -> FETCH_HEAD
 * [new branch]      master     -> origin/master
```



#### Unzip checkpoint file

내려받은 코드계층에서 `checkpoints`폴더 안에 있는 모델의 weights파일을 압축해제해줍니다.



```bash
$ cd checkpoints 
$ unzip ssd_300_vgg.ckpt.zip

>>> 
Archive:  ssd_300_vgg.ckpt.zip
  inflating: ssd_300_vgg.ckpt.data-00000-of-00001  
  inflating: ssd_300_vgg.ckpt.index 
```



#### installing dependency package



**Installing OpenCV**

python dependency package를 설치하기 이전에 먼저 opencv를 빌드합니다.

(만약 이미 Opencv가 Python3에서 사용할 수 있도록 빌드하였다면, 해당 내용은 생략해도 됩니다.)



1. Installing OpenCV dependency package

```bash
$ sudo apt-get update && apt-get install -y build-essential wget tar cmake git unzip yasm pkg-config libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libavformat-dev python3-dev python3-numpy python-dev python-numpy libgtk2.0-dev 
```



2. OpenCV Compile && Build

   해당 문서에서는 사용자가 python3.4 버전을 사용하며, OpenCV 3.4.2 버전을 컴파일한다는 것으로 가정하고 진행하였습니다.

   

   (해당 내용은 원하지 않을 경우, 변경하여도 됩니다.)

   ```bash
   $ wget https://github.com/opencv/opencv/archive/3.4.2.zip \
   && unzip 3.4.2.zip \
   && mkdir /opencv-3.4.2/cmake_binary \
   && cd /opencv-3.4.2/cmake_binary \
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
   ```

   ​    

3. OpenCV 설치 확인

   다음과 같은 명령어를 통해서 OpenCV 설치를 확인합니다.

   ```bash
   $ pkg-config --modversion opencv
   
   >>>
   3.4.2
   ```

   ​    

4. pyhon3에서 OpenCV Import 확인

   다음과 같은 명령어를 통해서 OpenCV가 Python3에서 import가 되는지 확인합니다.

   ```python
   import cv2
   
   >>
   Python 3.4 (default, Oct  3 2017, 21:45:48) 
   [GCC 7.2.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import cv2
   >>> 
   
   ```

   

**installing python dependency package**

SSD Implementation 코드를 실행할 수 있도록 python package를 다음과 같은 명령어를 이용하여 설치합니다.



```bash
$ pip3 install matplotlib pillow scipy numpy tensorflow==1.11.0 jupyter notebook
```

​    

### 3. Test SSD Tensorflow Implementation

jupyter notebook을 이용하여 설치한 SSD 구현체가 잘 작동하는지 확인합니다.



```bash
$ jupyter notebook ./notebooks 
```



![jupyter notebook list](https://user-images.githubusercontent.com/13328380/48117789-9d3c6280-e2ad-11e8-9c2a-1a009b58b240.png)

다음과 같은 jupyter notebook에서 `ssd_notebook.ipynb`를 클릭합니다.

클릭하면, 아래 그림과 같은 화면을 볼 수 있으며, 위의 `In [1]`의 오른쪽 박스를 클릭한 후, 차례차례 `shift + enter`키를 치면서 내려갑니다. 



(`shift + enter`를 누를 때, `In [*]`로 변경되는데, 이는 python이 해당 code block을 실행시키고 있는 것이므로, 해당 `*`이 숫자로 변경되기 전까지는 기다렸다가 `shift+enter`치는 것이 좋습니다.)



![Jupyter notebook display](https://user-images.githubusercontent.com/13328380/48117918-fa381880-e2ad-11e8-8dd6-295bace24405.png)



이때, 마지막 코드 블록에서 다음과 같은 화면이 뜬다면, SSD Tensorflow Implementation을 성공적으로 설치한 것입니다.



![final result](https://user-images.githubusercontent.com/13328380/48118124-a11cb480-e2ae-11e8-9f99-225b122a8107.png)

​    

## Wrapping SSD Prediction function

이제 위에서 설치한 SSD Tensorflow Implementation 코드를 DCF 함수 컴포넌트에서 사용할 수 있도록 추론`predict`코드만 랩핑해보도록 하겠습니다.



`ssd`폴더 안에 다음과 같은 추론`predict`코드인 `predict.py`를 작성합니다.

```bash
$ cd models/ssd
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

# optional
# import matplotlib
# matplotlib.use('Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

def predict(image):

    slim = tf.contrib.slim

    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)

    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    #reuse = True if 'ssd_net' in locals() else None
    reuse = tf.AUTO_REUSE
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model.
    ckpt_filename = 'models/ssd/checkpoints/ssd_300_vgg.ckpt'
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)


    # Main image processing routine.
    def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
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

    obj_info = dict()
    obj = list()
    for i in range(rclasses.shape[0]):
        cls_id = int(rclasses[i])
        if cls_id >= 0:
            score = rscores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(rbboxes[i, 0] * height)
            xmin = int(rbboxes[i, 1] * width)
            ymax = int(rbboxes[i, 2] * height)
            xmax = int(rbboxes[i, 3] * width)

            class_name = str(cls_id)

            obj_info["class"] = class_name
            obj_info["confidence"] = str(score)
            obj_info["xmin"] = str(xmin)
            obj_info["ymin"] = str(ymin)
            obj_info["xmax"] = str(xmin)
            obj_info["ymax"] = str(ymin)
            obj.append(obj_info)
    
    return obj

if __name__ == "__main__":
    print(predict(image=None))
```



해당 코드를 작성한 후에, `ssd-image`폴더로 들어와서, python script를 실행하면 다음과 같은 결과를 얻을 수 있습니다.



```bash
$ cd ../../
$ python3 models/ssd/prediction.py
```



![wrapping_result](https://user-images.githubusercontent.com/13328380/48118733-7f243180-e2b0-11e8-822b-23a325d78fa9.png)



​    

## Dockerfile

이제 본격적으로 DCF 함수 컴포넌트를 생성하기 위한 셋팅을 진행하도록 하겠습니다.

DCF 함수 컨테이너가 생성되면서 OpenCV를 설치할 수 있도록, Dockerfile을 다음과 같이 변경해줍니다.



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



​    

## handler.py

이제 base64 문자열을 입력으로하여 SSD 모델의 추론 결과를 JSON으로 돌려주는 DCF 함수 컴포넌트의 `handler.py`코드를 다음과 같이 작성해줍니다.



```python
from __future__ import print_function
import os
import sys
sys.path.append("./models/ssd/")
import io
import json
import base64
import numpy as np
import tensorflow as tf
import PIL

from PIL import Image
from models.ssd.predict import predict

def Handler(req):
    base64Str = req.input
    imageData = base64.decodestring(base64Str)
    img = Image.open(io.BytesIO(imageData))
    width, height = img.size
    
    img = np.array(img)

    result = predict(img)
    result = json.dumps(result)  
    
    return result
```

​    

## requirements.txt

DCF 함수 컴포넌트에 설치되어야하는 python dependency package를 `requirements.txt`에 명시해줍니다.



```bash
numpy
pillow
matplotlib
scipy
tensorflow==1.11.0
```

​    

## Deploy

이제 모든 준비가 끝났으므로, 다음과 같은 명령어를 이용하여 DCF CLI를 통해서 DCF 함수 컴포넌트를 생성해줍니다.



```bash
$ cd ../
$ sudo dcf function create -f config.yaml -v

>>>
Building: ssd-image, Image:keti.asuscomm.com:5001/ssd-image
Sending build context to Docker daemon    327MB
Step 1/25 : ARG PYTHON_VERSION=3.4
Step 2/25 : FROM python:${PYTHON_VERSION}
.
.
.
```

​    

## Invoke DCF Function

Deploy가 완료되었다면, 다음과 같은 명령어를 통해서 이미지를 ssd-image 함수에 입력으로 넣어서, 해당 이미지에 있는 객체의 `클래스`, `클래스의 확률`, `이미지에서의 객체 위치 좌표`를 반환받을 수 있습니다.



```bash
$ cat ssd-image/models/ssd/demo/000001.jpg | base64 | dcf function call ssd-image

>>>
[{"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}, {"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}]
```

​    
