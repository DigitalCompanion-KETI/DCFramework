# Digital Companion Framework (DCF)

해당 문서는 디지털동반자 프레임워크에 대해서 설명하며, 이를 어떻게 사용하는지 가이드 문서입니다.



## DCF 소개



DCF는 FaaS(Function as a Service)의 구조를 따릅니다. 다양한 기관들은 DCF를 이용하여 규격화된 인공지능 모델을 배포할 수 있습니다. 아래 그림은 DCF의 구조에 대한 간략화된 설명입니다.



각 기관의 개발자들은 `DCF CLI` 를 이용하여 인공지능 모델을 규격화하고, 배포할 수 있습니다. 규격화된 인공지능 모델은 Docker기반으로 배포됩니다.



이렇게 규격화된 인공지능 모델은 오른쪽에 보이는 하나의 Function이 될 수 있으며, 일반 사용자(유저) 및 DCF를 이용해 상위 어플리케이션을 개발하려는 개발자들은 각 기관이 배포한 Function을 Call하는 것을 통해서 인공지능 모델의 추론 결과를 얻을 수 있습니다.



![DCF-concept](https://user-images.githubusercontent.com/13328380/47892857-590c2500-de9d-11e8-8989-7821892b1a72.png)



#### Reference

[1. Apache OpenWhisk - 소개 및 아키텍쳐](https://developer.ibm.com/kr/cloud/2017/12/24/apache-openwhisk-intro-architecture/)

[2. (번역) 서버리스 아키텍처](https://blog.aliencube.org/ko/2016/06/23/serverless-architectures/)



​    

## DCF 설치



여기서 DCF를 설치한다는 의미는, DCF CLI를 설치한다는 의미가 됩니다. 

우리는 DCF CLI를 통해서 모든것(인공지능 모델 규격화, 배포)를 진행하게됩니다.



### 1. Docker 설치

DCF CLI를 설치하기 전에, 먼저 해당 컴퓨터에 Docker가 설치되어있어야합니다.

도커 버전은 17.05-CE버전 이상을 요구합니다.

설치 방법은 [링크](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)의 `Install Docker CE`를 참조하시면 됩니다.

​    

- docker >= 17.05-ce

​    

### 2. DCF CLI 다운로드

[DCF github](https://github.com/DigitalCompanion-KETI/DCFramework)을 참고하여, DCF CLI를 다운받고, 해당 파일의 권한을 수정합니다.

 

```bash
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v0.1.0/dcf
$ chmod +x dcf
```

​    

### 3. DCF CLI 설정

DCF 저장소에 로그인하기 위해서 다음과 같이 설정을 해줍니다.



```bash
$ echo '{"insecure-registries": ["keti.asuscomm.com:5001"]}'>> /etc/docker/daemon.json
$ service docker restart
```

```bash
$ docker login keti.asuscomm.com:5001
Username: elwlxjfehdqkswk
Password: elwlxjfehdqkswk
```



설정완료가 다 되면, 다음과 같은 명령어로 설정이 잘 되었는지 확인할 수 있습니다.



```bash
$ sudo docker info
>>
Insecure Registries:
 keti.asuscomm.com:5001
```

​    

## "Hello DCF"

   

이제 DCF를 통해서, 각 함수 컴포넌트들을 생성하고 배포, 테스트, 삭제까지하는 방법을 설명드리도록 하겠습니다.

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

다음과 같은 명령어로 Python3를 runtime으로 갖는 function을 정의할 수 있다.



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



