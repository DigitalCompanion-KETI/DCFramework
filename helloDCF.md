## "Hello DCF"

   

해당 가이드에서는 DCF를 통해서, 각 함수 컴포넌트들을 생성하고 배포, 테스트, 삭제까지하는 방법을 설명하도록 하겠습니다.

​        

### 1. Runtime 지원

DCF의 Runtime 지원 목록은 다음과 같은 명령어로 확인할 수 있습니다.

Runtime 지원이란, 컴포넌트의 function을 어느 언어로 작성할 것이냐에 대한 의미로 이해하면 됩니다.



```bash
$ ./dcf runtime list

Supported Runtimes are:
- python3
- tensorflow
- tensorflow-gpu
- go
- python2
```

​    

### 2. 컴포넌트 생성

다음과 같은 명령어로 Python3를 runtime으로 갖는 function을 정의할 수 있습니다.



```bash
$ ./dcf function init --runtime python3 <function name>

ex> ./dcf function init --runtime python3 helloDCF
```



위 명령어로 컴포넌트를 정의했다면, 다음과 같은 파일 구조를 확인할 수 있습니다.



```bash
<function name>
├── Dockerfile
├── handler.py
└── requirements.txt
```



- Dockerfile : 해당 함수의 Docker 컨테이너를 정의합니다.
- handler.py : DCF에 들어오는 요청이 들어오고 처리되는 스크립트입니다.
- requirements.txt : 해당 함수의 package dependency를 명시하는 파일입니다.

​    

### 3. 컴포넌트 배포

다음과 같은 명령어를 이용하여 정의한 컴포넌트를 DCF에 배포할 수 있습니다.

만약 배포되는 일련의 과정을 확인하고 싶다면, 해당 명령어 뒤에 `-v`을 추가합니다.



```bash
$ ./dcf function create -f config.yaml
Building: <function name>, Image:keti.asuscomm.com:5001/<function name>
Pushing: <function name>, Image:keti.asuscomm.com:5001/<function name>
Deploying: <function name>
```

​    

### 4. 배포 확인

다음과 같은 명령어를 이용하여 DCF에 컴포넌트가 잘 배포되어있는지 확인할 수 있습니다.

- `Status`가 Ready라면 해당 컴포넌트를 호출할 수 있습니다.
- `<Function name>` 은 중첩되면 안됩니다.



```bash
$ ./dcf function list

Function       	Image               	Maintainer     	Invocations	Replicas  	Status   
<function name>	$(repo)/<function name>    	               	0         	1      	Ready 
```

​    

### 5. 컴포넌트 호출

다음과 같은 명령어를 이용하여 DCF의 컴포넌트를 호출할 수 있습니다.



```bash
$ ./dcf function call echo-service 
$ echo "Hello, DCF!" | ./dcf function call echo-service 

Hello, DCF
```



handler.py를 확인해보면 다음과 같이 구성되어있는 것을 확인할 수 있습니다.

위의 예제를 설명해보면, `"Hello, DCF!"`라는 입력을 받으면 이를 그대로 return하는 구조로 볼 수 있습니다.

```python3
def Handler(req):
    return req
```

​    

### 6. 컴포넌트 삭제

 배포된 컴포넌트를 삭제하고 싶다면, 다음과 같은 명령어를 이용하여 삭제할 수 있습니다.

해당 함수삭제 여부는 배포 확인때와 같이 `./dcf function list`를 통해 확인할 수 있습니다. 



```bash
./dcf function delete <function name>
```



