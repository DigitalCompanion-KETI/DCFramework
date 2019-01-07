# Python Client Example

해당 가이드에서는 Python을 이용하여 DCF Client 프로그램을 작성하는 방법에 대해서 설명합니다.



여기서 사용하는 function은 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)로 진행합니다.

​    

## 1. 의존성 패키지 설치

아래와 같은 명령어를 이용하여 DCF 의존성 패키지를 설치합니다.



```bash
$ pip install -U protobuf
$ pip install dcfgrpc
```

​    

## 2. Client.py

기본적인 `echo-test` function에 대한 Client의 포맷은 다음과 같습니다.

```python
from dcfgrpc.api import dcf

if __name__ == '__main__':
    result = dcf(url='keti.asuscomm.com:32222',service="echo-test", arg="hello world".encode())
    print(result)
```

​    

이를 응용하면 적은 코드로 [3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)함수를 손쉽게 이용할 수 있습니다.



[3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)의 Client 예제는 다음과 같습니다.

```python
import sys
import io
import base64
from dcfgrpc.api import dcf

if __name__ == '__main__':
    img = Image.open('image file path')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    result = dcf(url='keti.asuscomm.com:32222',service="ssd-image", arg=img_str)
    print(result)
```



client.py을 작성 완료했다면 다음과 같은 형태로 테스트할 수 있습니다.

```bash
$ python client.py
[{"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}, {"ymin": "233", "xmin": "49", "ymax": "233", "xmax": "49", "class": "12", "confidence": "0.9948125"}]
```

​    
