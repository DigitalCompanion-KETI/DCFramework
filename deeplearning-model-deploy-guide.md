# Deeplearning model deploy Guide 

이 가이드는 딥러닝 모델을 DCF를 통해 배포하기 위한 가이드이다. 
해당 가이드문서는 Object Detection모델을 이용하여 이미지 상의 Object의 Class, Box, Confidence Score를 JSON Format으로 Return 하는 예제이다.

배포되는 서비스 예제는 다음과 같은 흐름을 따른다.

1. Image를 base64로 인코딩 (Client)
2. base64로 받은 Image 데이터를 디코딩(Server)
3. 디코딩 된 데이터를 Image로 변환(Server)
4. 변환된 Image를 입력으로 Model 추론
5. 추론 결과를 JSON으로 변환 및 리턴 


### PREREQUISITES 

- [SSD Tensorflow 구현체(Tensorflow 1.11.0)](https://github.com/balancap/SSD-Tensorflow)

### Create Python3 function  

function name이 중복되면 안되므로, DCF CLI를 이용하여 현재 배포되어있는 function name을 확인한다.
(만약 생성하려는 이름의 function이 deploy되어있다면, function name을 변경한다.)
```bash
$ dcf function init --runtime python3 <function name>

ex>
$ dcf function init --runtime python3 ssd-test
```

### Install SSD 구현체  
해당 가이드에서는 대표적인 딥러닝 기반의 Object Detection모델인 SSD의 Tensorflow 구현체를 이용하여 예제를 진행한다.
이를 위해 먼저 PREREQUISITES의 SSD Tensorflow 구현체를 설치하고 실행한다.

```bash
# 만든 함수의 경로로 들어가서, 추가 폴더를 생성한다.
cd <function directory>
mkdir models
mkdir ssd

# SSD tensorflow 구현체를 다운받는다.
git init
git remote add origin https://github.com/balancap/SSD-Tensorflow.git
git pull origin master

# weights 파일의 압축을 풀어준다.
cd checkpoints
unzip ssd_300_vgg.ckpt.zip
```

### Setting Dockerfile
SSD 구현체는 OpenCV에 의존성을 가지고 있으며, Python에서 OpenCV를 제대로 사용하려면, 빌드를 해야한다.
해당 가이드에서는 이미 빌드가 완료되어있는 OpenCV Docker Image를 사용한다.

함수의 폴더에서 Dockerfile을 다음과 같이 수정한다.

```Dockerfile
ARG REGISTRY
ARG WATCHER_VERSION=0.1.0

FROM ${REGISTRY}/watcher:${WATCHER_VERSION}-tensorflow as watcher
FROM yoanlin/opencv-python3:latest

RUN apt-get update \ 
    && apt-get install -y libgtk2.0-dev \
    libglib2.0-0 \
    build-essential

ARG handler_file=handler.py
ARG handler_name=Handler

ENV HANDLER_DIR=/dcf/handler
ENV HANDLER_FILE=${HANDLER_DIR}/${handler_file}
ENV HANDLER_NAME=${handler_name}

COPY --from=watcher /dcf/watcher ${HANDLER_DIR}
# RUN mkdir -p ${HANDLER_DIR}
WORKDIR ${HANDLER_DIR}
COPY . .
RUN touch ${HANDLER_DIR}/__init__.py

RUN pip install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Run?
# RUN cp -r /dcf/watcher/* ${HANDLER_DIR}

HEALTHCHECK --interval=1s CMD [ -e /tmp/.lock ] || exit 1

ENTRYPOINT ["python3"]
CMD ["server.py"]
```

### write new python script for DCF  
기존의 모델에서 inference코드를 따로 호출해서 사용할 수 있도록 만들어준다.
해당 가이드에서는 predict.py라는 스크립트를 SSD 구현체 폴더에서 다음과 같이 만든다.

```python
import sys
import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from models.ssd.nets import ssd_vgg_300, ssd_common, np_methods
from models.ssd.preprocessing import ssd_vgg_preprocessing
from models.ssd.notebooks import visualization

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

    # Test on some demo image and visualize output.
    #path = 'models/ssd/demo/'
    #image_names = sorted(os.listdir(path))

    #img = mpimg.imread(path + image_names[-5])


    rclasses, rscores, rbboxes =  process_image(image)

    # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

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
    print(predict())
```

### modify handler.py 
handler.py를 다음과 같이 수정해준다.

```python
from __future__ import print_function
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

### deploy function
```bash
dcf function create -f config.yaml -v
```
### function status check    
배포된 함수의 상태를 확인한다. 상태가 Ready가 되면, invoke or call 옵션을 이용하여 함수를 테스트한다.

```bash
watch dcf fucntion list
```

### invoke function

특정한 Image를 이용해 다음과 같이 함수를 호출한다.

```bash
cat 000001.jpg | base64 | ./dcf function call ssd-test

>>>
[{"class": "12", "confidence": "0.9948125", "xmin": "49", "ymin": "233", "xmax": "49", "ymax": "233"}, {"class": "12", "confidence": "0.9948125", "xmin": "49", "ymin": "233", "xmax": "49", "ymax": "233"}]
```

### modification
함수 수정을 원할 경우, 로컬에서 함수를 수정한 후, 다음과 같은 절차를 거쳐 재배포한다.

1. 함수를 delete한다.
```bash
dcf function delete <function name>

# 삭제한 함수가 없어지는 것을 확인
watch dcf function list
```   

2. 함수를 재생성한다.
```bash
dcf function create -f config.yaml -v 
```

OR
(수정 시)
```bash
dcf function create -f config.yaml -v --update
```

