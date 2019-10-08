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
