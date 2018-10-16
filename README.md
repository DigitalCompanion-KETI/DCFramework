# DCF - CLI 

디지털 동반자에 지능 컴포넌트를 생성하기 위해서 DCF CLI를 사용한다. DCF CLI는 다양한 기관의 디지털 동반자  지능 컴포넌트를 규격화하여 공통으로 사용 가능하도록 자율지능 디지털 동반자 지능 컴포넌트 구조를 만들어 준다. 

## Requirement
- docker version >= 17.05-ce

## Get started: Install the CLI 

```
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v0.1.0/dcf
$ chmod +x dcf
```

## Prerequisites for CLI

```
$ echo '{"insecure-registries": ["keti.asuscomm.com:5001"]}'>> /etc/docker/daemon.json
$ service docker restart
```
디지털 동반자 레포지토리에 로그인 하기 위해서 'docker login' 명령과 함께 다음과 같은 임시 아이디와 비밀번호를 입력한다.

```
$ docker login keti.asuscomm.com:5001
Username: elwlxjfehdqkswk
Password: elwlxjfehdqkswk
```

## Run the CLI 


### 디지털 동반자 지능 컴포넌트(Function) 생성 
다음은 예를 들어 python 런타임 사용시에 절차를 나타낸다.

1. __지원되는 Runtime의 목록을 보여준다.__ 
  
    ```
    $ dcf runtime list
    ```
	> Runtime: DCF에서 지원해주는 Function 실행 환경
	
2. __Runtime을 지정하여 지능컴포넌트(function)를 정의한다.__
	예를 들어, echo-service라는 지능컴포넌트를 정의할 때 원하는 runtime을 flag를 통해 지정할 수 있다. 초기화(init)가 완료되면 현재 디렉토리에 설정파일(default: config.yaml)과 echo-service라는 폴더가 만들어지고, Python 런타임 사용시, 폴더 안에는 handler.py 파일과 Dockerfile, requirements.txt이 생성된다.
    ```
    $ dcf function init --runtime python3 echo-service
    ```
    설정파일의 예는 다음과 같다. 
    > 사용자가 지능컴포넌트를 정의하고자 할 때 규격이 되는 파일, 여러 개의 function을 정의한 yaml 파일
    
    ```
    dcf:
      gateway: keti.asuscomm.com:32222    
    functions:
      echo-service:
        runtime: python3
        desc: "This is echo service."
        maintainer: "KETI"
        handler:
          dir: ./echo-service
          file: handler.py
          name: Handler
        image: keti.asuscomm.com:5001/echo-service
    ```
    
    설정파일의 규격은 다음과 같다.
    
    | Field  | Description | Example | 
    |------------- |-------------|-------------| 
    |  | 지능 컴포넌트의 이름| echo-service|
    |runtime|지능 컴포넌트가 실행될 환경|python3|
    |image|지능 컴포넌트 이미지 이름과 버전<br>(레포지토리인 keti.ausscomm.com:5001은 고정)|keti.asuscomm.com:5001/echo-service:v1|
    |handler|지능 컴포넌트 배포시에 실행되는 엔트리포인트 정보|handler:<br>&nbsp; name: Handler<br>&nbsp; dir: ./echo-service<br>&nbsp; file: "handler.py"|
    |maintainer|(optional)지능 컴포넌트 개발자 또는 유지보수 가능한 사람의 정보|KETI|
    |desc|(optional)지능 컴포넌트 용도/설명|This is ....|
    |environment|(optional)런타임 내에서 사용할 환경 변수|environment:<br>&nbsp; - "PATH=/usr/local/bin"|
    |skip_build|(optional)지능 컴포넌트 빌드 및 레포지토리에 저장 단계 건너뛰기| skip_build: true|
    |limits|(optional)지능 컴포넌트가 사용할 자원 요청 및 제한| limits:<br>&nbsp; cpu: "1"<br>&nbsp; gpu: "1"<br>&nbsp; memory: "1G"|
    |build_args|(optional)Dockerfile내에 ARG 값 지정|build_args:<br>&nbsp; - "PYTHON_VERSION=3.7"|
    
	requirements.txt 수정을 통해 런타임 내에 설치할 python 패키지 버전을 명시한다. 
	```
	scapy==2.4.*
	tinyec>=0.3.1
	```
	handler.py 수정을 통해 지능컴포넌트의 main 함수를 정의한다.
	```
	def Handler(req):
	    return req.input
	```

3. __설정파일을 통해 지능컴포넌트를 생성한다.__
  
    ```
    $ dcf function create -f config.yaml
    Building: echo-service, Image:keti.asuscomm.com:5001/echo-service
    Pushing: echo-service, Image:keti.asuscomm.com:5001/echo-service
    Deploying: echo-service
    ``` 
### 지능컴포넌트(function) 확인 
  
생성된 지능 컴포넌트가 Ready 상태인지 확인

  ```
  $ dcf function list
  Function       	Image               	Maintainer     	Invocations	Replicas  	Status    	Description
  echo-service   	$(repo)/echo-service	KETI           	4         	1         	Ready     	This is echo service    
  ``` 

### 지능컴포넌트(function) 호출 

  ```
  $ dcf function call echo-service 
  $ echo "Hello,World!" | dcf function call echo-service
  $ cat "cat.png" | dcf function call echo-service
  $ echo "Hello,World!" | dcf function call echo-service | dcf function call echo-service2
  ```  



### 생성된 지능컴포넌트(function) 의 정보 확인

  ```
  $ dcf function info echo-service
  ```

### 지능컴포넌트(function) 삭제
디지털 동반자에 배포된 지능 컴포넌트가 제거된다. 로컬의 설정 파일과 디렉토리는 제거되지 않는다. 필요시에는 수동으로 제거한다.

  ```
  $ dcf function delete -f config.yaml 
  $ dcf function delete echo-service
  ```
  
  ---
CLI 명령어에 대한 도움말은 다음의 Flag를 통해 실행할 수 있다. 

```
$ dcf function -h 
$ dcf function init -h 
$ dcf runtime -h 
$ dcf runtime list -h 
```

## 지능 컴포넌트 개발자를 위한 gRPC API 가이드
지능 컴포넌트 개발자를 위한 gRPC API 설치 및 사용법은 [gRPC Guide](https://github.com/DigitalCompanion-KETI/DCFramework/blob/master/grpc-guide.md)를 참조하길 바랍니다.
