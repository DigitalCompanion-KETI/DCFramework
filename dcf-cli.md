# DCF-CLI

DCF-CLI는 터미널을 이용해서 DCF와 다음과 같은 상호작용을 할 수 있다.

- 함수 런타임 목록 조회
- 함수 생성
- 함수 빌드
- 함수 테스트
- 함수 배포
- 함수 호출
- 함수 로그 확인



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

#### 1.2.1 Prerequiesites

##### 1.2.1.1 Golang

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

$ source ~/.bash
```



환경 변수 설정을 완료했다면 Go 설치를 확인한다.

```bash
$ go version
$ go env
```



#### 1.2.2 Compile

##### 1.2.2.1 DCF-CLI Clone from Github

```bash
$ cd $GOPATH/src/github.com
$ git clone https://github.com/digitalcompanion-keti/dcf-cli.git
# OR
$ go get https://github.com/digitalcompanion-keti/dcf-cli
```



##### 1.2.2.2 Build DCF-CLI

```bash
$ go build
$ go install
```



##  2 Inquire runtime list

```bash
$ dcf-cli runtime list
>>>
Supported Runtimes are:
- python
- go
```



## 3 Create function

```bash
$ dcf-cli function init [function name] --runtime [runtime] -f [name of configuration yaml file. default name is config.yaml] --gateway [dcf gateway address]
>>> dcf-cli function init echo --runtime python
Directory: echo is created.
Function handler created in directory: echo/src
Rewrite the function handler code in echo/src directory
Config file written: config.yaml
```



## 4 Write handler 

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



## 8 Invoke function

```bash
$ dcf-cli function list
$ echo "Hello DCF" | dcf-cli invoke [function name]
>>> echo "Hello DCF" | dcf-cli invoke echo
Hello, DCF
```



## 9 Log of function

```bash
$ dcf-cli function log [function name]
>>> dcf-cli function log echo
Error: did not get log: rpc error: code = Internal desc = the server rejected our request for an unknown reason (get pods echo-77446d455f-kpqlc)
```

