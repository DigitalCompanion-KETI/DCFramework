# Call Function for Python Lib

해당 예제는 `python` 으로 클라이언트를 생성하여 DCF 함수를 호출하는 예제이다.  또한, 통신할 DCF 함수를 서버에 미리 배포해야 하며 본 예제에서는 앞 예제 함수 중 `echo` 함수를 호출한다.  



### Install Python library

DCF 플랫폼 통신 라이브러리를 설치한다. 

```bash
$ pip install dcfgrpc
```

### Write code

`test.py`를 생성하여 다음의  코드를 작성한다.

```python
from dcfgrpc.api import dcf

if __name__ == '__main__':
    result = dcf(url='10.0.7.1:32222',service="echo", arg="hello world".encode())

    print(result)
```

DCF 통신을 위한 매개변수는 다음과 같다.

* `url` : 호출할 DCF 서버 와 포트를 작성한다.

* `service` : 호출할 DCF 함수를 작성한다.

* `arg`:호출할 함수에 전달할 메세지를 작성한다.  Byte array로 통신되므로 함수에서 문자열을 받고자 한다면, 예제처럼 메세지 뒤에 `.encode()`를 붙여야한다. 

### Test function

작성한 코드를 실행한다. 

```bash
$ python test.py
> hello world 
```
