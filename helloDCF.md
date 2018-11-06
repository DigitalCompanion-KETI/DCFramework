## "Hello DCF"

   

해당 가이드에서는 DCF를 통해서, 디지털 동반자 지능 컴포넌트(function)들을 생성하고 배포, 테스트, 삭제까지하는 방법을 설명하도록 하겠습니다.

​        

### 1. Runtime 지원

DCF의 Runtime 지원 목록은 다음과 같은 명령어로 확인할 수 있습니다.

> Note

Runtime 이란, DCF에서 지원해주는 지능 컴포넌트의 실행 환경 입니다. 

즉, 지능 컴포넌트를 어느 언어로 작성할 것이냐에 대한 의미로 이해하면 됩니다.



```bash
$ ./dcf runtime list

Supported Runtimes are:
- python
- tensorflow
- tensorflow-gpu
- go
```

- 지능 컴포넌트 생성 시, `dcf-runtime`이라는 디렉토리가 생성되고, 디렉토리 내부의 list.yml 파일에서 

  runtime의 버전에 대한 정보를 기입할 수 있습니다. 이는 유동적으로 runtime의 버전을 지원하기 위함입니다.

  ```bash
  >> list.yml

  runtimes:
    go:
      dir: go
      build_args:
        - GO_VERSION=1.10
      handler:
        name: Handler
    python:
      dir: python
      build_args:
        - PYTHON_VERSION=3.4
      handler:
        file: handler.py
        name: Handler
    tensorflow-gpu:
      dir: tensorflow-gpu
      handler:
        file: handler.py
        name: Handler
    tensorflow:
      dir: tensorflow
      handler:
        file: handler.py
        name: Handler
  ```

​    

### 2. 컴포넌트 생성

다음과 같은 명령어로 Python을 runtime으로 갖는 지능 컴포넌트(function)를 정의할 수 있습니다.

> Notify

- build_args, build_packages 필드가 추가된 config.yaml 파일을 적용하기 위해서 기존의 지능 컴포넌트 생성 시 

  생성된 dcf-runtime 디렉토리를 삭제하고 지능 컴포넌트를 새로 생성하여 주시기 바랍니다.

```bash
$ dcf function init --runtime python <function name>

ex> dcf function init --runtime python hello-dcf
ex> dcf function init --runtime go hello-dcf
```



위 명령어로 컴포넌트를 정의했다면, 다음과 같은 파일 구조를 확인할 수 있습니다.



```bash
>> Runtime: Python

├──config.yaml
<dcf-runtime>
├──list.yml
├──Makefile
├──<python>
├──<tensorflow>
├──<tensorflow-gpu>
└──<go>
<function name>
├── Dockerfile
├── handler.py
└── requirements.txt
```
---

```bash
>> Runtime: Go

├──config.yaml
<dcf-runtime>
├──list.yml
├──Makefile
├──<python>
├──<tensorflow>
├──<tensorflow-gpu>
└──<go>
<function name>
├── Dockerfile
├── handler.go
└── Gopkg.toml
```

> Note

Dockerfile 

- 해당 함수의 Docker 컨테이너를 정의합니다.

handler.py / handler.go

- DCF로 들어오는 요청을 처리해주는 스크립트입니다. 입력되는 요청을 실제로 처리해주는 함수를 정의할 수 있습니다. 

  아래의 예제에서는 입력 데이터를 string으로 반환해주는 함수입니다. 

  ```bash
  >> handler.py

  def Handler(req):
      return req.input
  ```
  
  ---
  
  ```bash
  >> handler.go
  
  package main

  import sdk "github.com/digitalcompanion-keti/dcf-watcher/go/pb"

  func Handler(req sdk.Request) string {
          return string(req.Input) 
  }
  ```

requirements.txt / Gopkg.toml

- 해당 함수의 package dependency를 명시하는 파일입니다. 이는 Runtime에 따라서 종속 패키지를 설치하기 위함이며, 

  다음과 같이 수정할 수 있습니다.

  ```bash
  >> requirements.txt

  tensorflow==1.11.0
  pillow
  matplotlib
  numpy
  scipy
  ```

  ---

  ```bash
  >> Gopkg.toml

  [[constraint]]
  name = "github.com/user/project"
  ```

config.yaml 
 
- 사용자가 지능 컴포넌트를 정의하고자 할 때 규격이 되는 파일이며 여러 개의 지능 컴포넌트를 정의할 수 있습니다. 

  ```bash
  >> config.yaml
    
  functions:
    hello-dcf:
      runtime: python
      desc: "This is Hello dcf."
      maintainer: "KETI"
      handler:
        dir: ./hello-dcf
        file: handler.py
        name: Handler
      image: keti.asuscomm.com:5001/hello-dcf
      build_args:
      - PYTHON_VERSION=3.4
      build_packages:
        - make
        - python3-pip
        - gcc
        - python-numpy
  dcf:
    gateway: keti.asuscomm.com:32222
  ```

- 설정 파일의 규격은 다음과 같습니다.

    | Field  | Description | Example | 
    |------------- |-------------|-------------| 
    | hello-dcf | 지능 컴포넌트의 이름| hello-dcf|
    |runtime|지능 컴포넌트가 실행될 환경|python3|
    |image|지능 컴포넌트 이미지 이름과 버전<br>(레포지토리인 keti.ausscomm.com:5001은 고정)|keti.asuscomm.com:5001/hello-dcf:v1|
    |handler|지능 컴포넌트 배포시에 실행되는 엔트리포인트 정보|handler:<br>&nbsp; name: Handler<br>&nbsp; dir: ./hello-dcf<br>&nbsp; file: "handler.py"|
    |maintainer|(optional)지능 컴포넌트 개발자 또는 유지보수 가능한 사람의 정보|KETI|
    |desc|(optional)지능 컴포넌트 용도/설명|This is ....|
    |environment|(optional)런타임 내에서 사용할 환경 변수|environment:<br>&nbsp; - "PATH=/usr/local/bin"|
    |skip_build|(optional)지능 컴포넌트 빌드 및 레포지토리에 저장 단계 건너뛰기| skip_build: true|
    |limits|(optional)지능 컴포넌트가 사용할 자원 요청 및 제한| limits:<br>&nbsp; cpu: "1"<br>&nbsp; gpu: "1"<br>&nbsp; memory: "1G"|
    |build_args|Dockerfile내의 ARG 값으로 지정|build_args:<br>&nbsp; - "PYTHON_VERSION=3.4"|
    |build_packages|Dockerfile내의 apt 패키지 관리자인 ADDITIONAL_PACKAGE 값으로 설정|build_packages:<br>&nbsp; -make<br>&nbsp; -python3-pip<br>&nbsp; -gcc<br>&nbsp; -python-numpy|

> Note

만약 hello-dcf라는 지능 컴포넌트가 존재하는 경우 다음과 같은 error가 발생할 수 있습니다. 

이와 같은 경우에는 지능 컴포넌트의 이름을 다른 이름으로 지정하여 컴포넌트를 생성합니다.

```bash
Error: Function hello-dcf already exists in config.yaml file.
```

​    

### 3. 컴포넌트 배포

다음과 같은 명령어를 이용하여 config.yaml 파일을 통해 정의한 컴포넌트를 DCF에 배포할 수 있습니다.

만약 배포되는 일련의 과정을 확인하고 싶다면, 해당 명령어 뒤에 `-v`을 추가합니다.



```bash
$ dcf function create -f config.yaml
Building: <function name>, Image:keti.asuscomm.com:5001/<function name>
Pushing: <function name>, Image:keti.asuscomm.com:5001/<function name>
Deploying: <function name>
```

​    

### 4. 배포 확인

Kubernetes에서는 비동기 처리를 하기 때문에 대기열에서 응답을 기다리기 때문에 지능컴포넌트를 생성하고 

시간이 지난 후에 해당 지능컴포넌트 Image의 Status가 Ready 상태로 변경됩니. 

지능컴포넌트를 생성한 시점에서는 Not Ready 상태이고 이 때 지능컴포넌트를 호출하면 다음과 같은 error가 발생합니다.

```bash
rpc error: code = Internal desc = rpc error: code = DeadlineExceeded desc = context deadline exceeded
```

따라서 다음과 같은 명령어를 이용하여 DCF에 컴포넌트가 잘 배포되어있는지, 즉 컴포넌트의 Status가 Ready 인지를 확인하여야 합니다.

```bash
$ dcf function list

Function       	Image               	Maintainer     	Invocations	    Replicas    Status     Description
<function name>	$(repo)/<function name>    	         0         	    1           Ready 
```

​    

### 5. 컴포넌트 호출

다음과 같은 명령어를 이용하여 DCF의 컴포넌트를 직접 호출하거나 파이프라이닝을 통해 입력 값을 

전달하여 호출할 수 있습니다.



```bash
$ dcf function call hello-dcf 
$ echo "Hello, DCF!" | dcf function call hello-dcf 

Hello, DCF
```

​    

### 6. 컴포넌트 삭제

 배포된 컴포넌트를 삭제하고 싶다면, 다음과 같은 명령어를 이용하여 삭제할 수 있습니다.

해당 컴포넌트 삭제 여부는 배포 확인때와 같이 `dcf function list`를 통해 확인할 수 있습니다.

컴포넌트의 Status가 Ready에서 Not Ready로 변경되고 시간이 지나면 해당 컴포넌트는 목록에서 없어지게 됩니다.



```bash
$ dcf function delete <function name>
$ dcf function list
Function       	Image               	Maintainer     	Invocations	    Replicas    Status     Description
<function name>	$(repo)/<function name>    	         0         	    1           Not Ready 

$ dcf function list
Function       	Image               	Maintainer     	Invocations	    Replicas    Status     Description
```



