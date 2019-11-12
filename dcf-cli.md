# DCF-CLI

DCF-CLI는 터미널을 이용해서 DCF와 다음과 같은 작업을 할 수 있다.

- 함수 런타임 목록 조회
- 함수 생성
- 함수 빌드
- 함수 테스트
- 함수 배포
- 함수 호출
- 함수 로그 확인

## prerequesit

플랫폼 구동 환경(CPU, GPU)의 설치 사항은 다음의 표에서 확인 할 수 있습니다. 

| CPU                                                                                                                                                                            | GPU                                                                                                                                                                                                                                                                           |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [Docker 19.03 설치](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/) | [Docker 19.03 설치](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/), [Nvidia-Docker 2 설치](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) |

또한, local nvidia-docker2에서 `GPU`를 구동하기 위해선 아래와 같이 확인 사항이 필요합니다. 

* [CUDA toolkit 을 사용하기 위해 요구되는 최소 Driver 버전과 GPU 아키텍쳐 요구버전 확인](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements)
  
  | CUDA toolkit version | Driver version         | GPU architecture                                                                                        |
  | -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------- |
  | 6.5                  | >= 340.29              | >= 2.0 (Fermi)                                                                                          |
  | 7.0                  | >= 346.46              | >= 2.0 (Fermi)                                                                                          |
  | 7.5                  | >= 352.39              | >= 2.0 (Fermi)                                                                                          |
  | 8.0                  | == 361.93 or >= 375.51 | == 6.0 (P100)                                                                                           |
  | 8.0                  | >= 367.48              | >= 2.0 (Fermi)                                                                                          |
  | 9.0                  | >= 384.81              | >= 3.0 (Kepler)                                                                                         |
  | 9.1                  | >= 387.26              | >= 3.0 (Kepler)                                                                                         |
  | 9.2                  | >= 396.26              | >= 3.0 (Kepler)                                                                                         |
  | 10.0                 | >= 384.111, < 385.00   | [Tesla GPUs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#flexible-upgrade-path) |
  | 10.0                 | >= 410.48              | >= 3.0 (Kepler)                                                                                         |
  | 10.1                 | >= 384.111, < 385.00   | [Tesla GPUs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#flexible-upgrade-path) |
  | 10.1                 | >=410.72, < 411.00     | [Tesla GPUs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#flexible-upgrade-path) |
  | 10.1                 | >= 418.39              | >= 3.0 (Kepler)                                                                                         |

## 1 Installation

DCF-CLI를 설치하는 방법은 아래와 같이 두가지 방법으로 설치할 수 있다.

- 컴파일 되어있는 바이너리 파일 다운로드
- 소스코드로 부터 다운로드

### 1.1 Download Binary

```bash
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v1.0.0/dcf-cli
$ mv dcf-cli /usr/bin
```

### 1.2 Compile from source

#### 1.2.1 Prerequisites

##### Golang

[공식 Go 다운로드 홈페이지](https://golang.org/doc/install)에서 자신의 환경에 맞게 설치파일을 다운로드 받는다.

설치파일을 압축 해제하고, `go` 폴더를 `/usr/local`로 옮긴다.

```bash
$ sudo tar -xvf go1.12.5.linux-amd64.tar.gz
$ sudo mv go /usr/local
```

`~/.bashrc`파일을 수정하여 환경변수 설정을 진행 및 적용한다

```bash
$ vim ~/.bashrc
>>
# add this lines
export GOROOT=/usr/local/go
export GOPATH=$HOME/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

$ source ~/.bashrc
```

환경 변수 설정을 완료했다면 Go 설치를 확인한다.

```bash
$ go version
$ go env
```

#### 1.2.2 Compile

##### 1.2.2.1 pb Clone from Github

먼저 해당 디렉토리가 없는지 확인하고 디렉토리를 생성한다.

```bash
$ mkdir -p $GOPATH/src/github.com/digitalcompanion-keti
```

디지털 동반자 프레임워크의 pb 저장소를 다운받는다.

```bash
$ cd $GOPATH/src/github.com/digitalcompanion-keti
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v1.0.0/pb-master.zip
$ unzip pb-master.zip
$ mv pb-master pb
```

##### 1.2.2.2 DCF-CLI Clone from Github

디지털 동반자 프레임워크 DCF-CLI 저장소를 다운받는다.

```bash
$ cd $GOPATH/src/github.com/digitalcompanion-keti
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v1.0.0/dcf-cli-master.zip
$ unzip dcf-cli-master.zip
$ mv dcf-cli-master dcf-cli
```

##### 1.2.2.3 Build DCF-CLI

아래 명령어를 이용하여 DCF-CLI를 빌드한다.

```bash
$ cd dcf-cli
$ go build
$ go install
```

### 1.3 Docker Private Registry Configuration

디지털 동반자 프레임워크의 도커 저장소에 로그인하기 위한 설정을 진행한다

#### 1.3.1 Insecure Registry

 `daemon.json`파일을 아래와 같이 작성한다

```bash
$ sudo vim /etc/docker/daemon.json
```

```json
{
    "insecure-registries": ["keti.asuscomm.com:5001"]
}
```

작성한 후 Docker를 재시작한다

```bash
$ sudo service docker restart
```

Insecure registry 등록이 잘 되어있는지 아래 명령어로 확인한다

```bash
$ sudo docker info
>>
Insecure Registries:
 keti.asuscomm.com:5001
```

#### 1.3.2 Docker Login

디지털 동반자 프레임워크 도커 저장소에 로그인한다

```bash
$ docker login keti.asuscomm.com:5001
>>
Username: elwlxjfehdqkswk
Password: elwlxjfehdqkswk
```

## 2 Inquire runtime list

디지털 동반자 프레임워크는 함수 런타임으로 Python과 GO를 지원한다. 

DCF-CLI를 이용하여 디지털 동반자 프레임워크에서 지원하는 런타임 리스트를 조회할 수 있다.

```bash
$ dcf-cli runtime list
>>>
Supported Runtimes are:
- python
- go
```

## 3 Create function

디지털 동반자 프레임워크에서 사용할 함수는 DCF-CLI의 아래 명령어를 이용해서 만들 수 있다. 

```bash
$ dcf-cli function init [function name] --runtime [runtime] -f [name of configuration yaml file. default name is config.yaml] --gateway [dcf gateway address]
>>> dcf-cli function init echo --runtime python
Directory: echo is created.
Function handler created in directory: echo/src
Rewrite the function handler code in echo/src directory
Config file written: config.yaml
```

## 4 Write handler

DCF-CLI를 이용하여 함수를 만들었다면 `src/handler.py`에  `Handler`라는 클래스가 작성되어있음을 확인할 수 있다. 사용자 정의 함수는 `Handler`클래스의 내부에 작성한다.

```bash
$ cd echo/src
$ vim handler.py
>>>
class Handler:
    def __init__(self):
        pass

    def __call__(self, req):
        return req.input
```

## 5 Build function

작성한 함수는 DCF-CLI의 `build`명령어로 도커 이미지로 빌드할 수 있다. `-v`옵션을 사용하면 도커 이미지가 빌드되면서 출력하는 메세지를 확인할 수 있다.

```bash
$ cd echo
$ dcf-cli function build -f [name of configuration yaml file] -v
Building function (echo) ...
Sending build context to Docker daemon  8.192kB
Step 1/45 : ARG ADDITIONAL_PACKAGE
Step 2/45 : ARG REGISTRY
Step 3/45 : ARG PYTHON_VERSION
Step 4/45 : ARG GRPC_PYTHON_VERSION=1.4.0
Step 5/45 : ARG WATCHER_VERSION=0.1.0
Step 6/45 : ARG handler_file=handler.py
Step 7/45 : ARG handler_name=Handler
Step 8/45 : ARG handler_dir=/dcf/handler
Step 9/45 : ARG handler_file_path=${handler_dir}/src/${handler_file}
Step 10/45 : ARG CUDA_VERSION=9.0
Step 11/45 : ARG CUDNN_VERSION=7.4.1.5
Step 12/45 : ARG UBUNTU_VERSION=16.04
Step 13/45 : ARG CUDA_VERSION_BACKUP=${CUDA_VERSION}
...
```

## 6 Test function

빌드한 도커 이미지를 디지털 동반자 프레임워크 환경에 배포하기 전에 `run` 명령어를 사용해서 사용자의 컴퓨팅 환경에서  테스트할 수 있다. 이를 활용해 배포하기 전에 작성한 함수가 올바르게 작동하는지 부분적으로 검증할 수 있다.

```bash
$ echo "Hello DCF" | dcf-cli function run [Function name]
>>> echo "Hello DCF" | dcf-cli function run echo
Checking VGA Card...
01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GM206GL [Quadro M2000] [10de:1430] (rev a1)
01:00.1 Audio device [0403]: NVIDIA Corporation Device [10de:0fba] (rev a1)


Checking Driver Info...

==============NVSMI LOG==============

Timestamp                           : Mon Oct  7 14:17:07 2019
Driver Version                      : 390.116

Attached GPUs                       : 1
...
Running image (keti.asuscomm.com:5001/echo) in local
Starting [dcf-watcher] server ...
Call echo in user's local
Handler request: Hello

Handler reply: Hello
```

## 7 Deploy function

사용자 컴퓨터 환경에서 함수를 테스트했다면, `deply`라는 명령어를 이용해서 디지털 동반자 프레임워크 환경에 함수를 배포할 수 있다. `-v`옵션을 사용하면 만든 도커 이미지가 디지털 동반자 프레임워크의 도커 레지스트리 서버로 전송하며 출력하는 메세지를 확인할 수 있다.

```bash
$ cd echo
$ dcf-cli function deploy -f config.yaml -v
>>>
Is docker registry(registry: keti.asuscomm.com:5001) correct ? [y/n] y
Pushing: echo, Image: keti.asuscomm.com:5001/echo in Registry: keti.asuscomm.com:5001 ...
The push refers to repository [keti.asuscomm.com:5001/echo]
519b1665e7d6: Preparing 
913823b0a3b0: Preparing 
abcdb1a22c59: Preparing 
10b48c649b87: Preparing 
3d2effb69d5a: Layer already exists 
8e9de3569873: Waiting 
...
```

## 8 Function list

함수를 배포했다면 `list`라는 명령어를 이용하여 디지털 동반자 프레임워크에 배포되어있는 함수를 확인할 수 있다. 함수의 상태(Status)가 준비(Ready) 상태라면 함수를 호출할 수 있다. 만약 함수가 긴 시간동안 준비되지 않음(Not Ready)를 유지한다면 **10. Log of function**을 참고하여 함수가 배포되지 않는 이유를 확인할 수 있다.

```bash
$ dcf-cli function list
Function           Image                   Maintainer         Invocations    Replicas      Status        Description                             
echo               $(repo)/echo                               0             1             Ready
```

## 9 Invoke function

함수가 배포되어 호출할 준비가 되었다면 아래의 명령어를 이용하여 함수를 호출할 수 있다.

```bash
$ dcf-cli function list
$ echo "Hello DCF" | dcf-cli function invoke [function name]
>>> echo "Hello DCF" | dcf-cli function invoke echo
Hello, DCF
```

## 10 Log of function

배포된 함수가 올바르게 작동하지 않는다면 `log`명령어를 사용하여 함수의 로그 정보를 확인할 수 있다.

```bash
$ dcf-cli function log [function name]
>>> dcf-cli function log echo
Error: did not get log: rpc error: code = Internal desc = the server rejected our request for an unknown reason (get pods echo-77446d455f-kpqlc)
```
