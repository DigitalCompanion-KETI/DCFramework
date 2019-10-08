## Bi-Directional Streaming Example for Video



본 예제는 `User Client`에서 웹 캠 또는 동영상 파일을 입력받아 `Handler.py`와 동영상 데이터를 주고 받는 예제이다.  `User client` 와 `Handler.py`의 수행 역할은 다음과 같다.

* `Handler.py`, DCF 플랫폼 내에서 처리되는 함수이며 DCF Client 에서 발신한 데이터에 대해 동영상 처리를 수행한 후 DCF Client로 재전송한다.
- `User Client`, 데이터 입력 및 통신을 위한 Bytes 변환을 수행한다.



## Handler.py

스트리밍 통신을 확인하기 위해 함수를 생성한다.  

### Init

streaming 함수를 생성한다.

```bash
$ dcf-cli function init streaming --runtime streaming
>>
Directory: streaming is created.
Function handler created in directory: hello-dcf/src
Rewrite the function handler code in hello-dcf/src directory
Config file written: config.yaml
```



### Write function

#### handler.py

아래와 같이 `handler.py`를 작성한다.

```python
import numpy as np 
import cv2 

def Handler(req):
    # Bytes -> frame 
    nparr = np.frombuffer(req.input, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    
    """
    frame 데이터 처리 
    """

    # Frame -> Bytes
    res = cv2.imencode('.jpg', frame)[1].tostring()

    return res
```

#### requirements.txt

다음은 데이터 변환에 필요한  패키지 파일을 requirements.txt에 명시한다. 

```textile
opencv-python
opencv-contrib-python
ffmpeg
```

### Build function

작성한 함수를 빌드한다

```bash
$ cd streaming
$ dcf-cli function build -f config.yaml -v
>>
Building function (streaming) ...
Sending build context to Docker daemon  8.192kB
Step 1/45 : ARG ADDITIONAL_PACKAGE
Step 2/45 : ARG REGISTRY
Step 3/45 : ARG PYTHON_VERSION
...
```









## User Client

#### Init

`User Client`는  Python 언어로 구현하였으며 필요 라이브러리는 다음의 명령어를 통해 설치할 수 있다. 비디오 데이터 변환 및 입력을 위한 라이브러리로 Opencv를 사용하였다.

```cmd
pip install opencv-python
pip install opencv-contrib-python
pip install ffmpeg 

python -m pip install grpcio
python -m pip install grpcio-tool

pip install argparse
```

*"Opencv 외 라이브러리 통해 데이터 인코딩 및 입력이 가능하지만, Handler 함수에서 사용자 라이브러리 설치 및 데이터 디코딩이 필요하다."*



#### Test

Client 를 실행하기 위한 명령어는 다음과 같다.

```python
$ python client.py -h
> 

This code is written for DCT Client about Bi-Streaming for Video

positional arguments:
  DCF Function name  Input to Use DCF Function
  Video file Name    Input to Use Video File Name if you use webcam, Just
                     input 0

optional arguments:
  -h, --help         show this help message and exit
```

```python
$ python3 client.py [$function] --video [$video File]
```

- [$function] : 사용할 DCF Handler 함수를 등록한다.

- [$Vdieo File] : 사용할 동영상 파일명을 등록한다. 동영상 경로는 현 실행 폴더로 지정해뒀다. 또한 웹 캠으로 동영상 데이터를 입력받을 시 `0`을 입력한다.

다음 명령어는 웹 캠에서 입력받은 동영상 데이터를 streaming 함수로 처리하는 명령어이다.

```python
$ python3 client.py streaming 0 
```

![dcf-streaming](https://user-images.githubusercontent.com/46108451/66360467-31c7ad00-e9b6-11e9-9e8a-bd34b769aab0.PN)![dcfstreaming](https://user-images.githubusercontent.com/46108451/66360467-31c7ad00-e9b6-11e9-9e8a-bd34b769aab0.PNG)

다음 명령어는 폴더내 동영상을 읽어 streaming 함수로 처리하는 명령어이다.

```python
$ python client.py streaming fancy.mp4
```
