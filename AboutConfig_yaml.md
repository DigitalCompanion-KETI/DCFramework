# About config.yaml



해당 문서는 함수 컴포넌트를 생성할 때, 만들어지는 `config.yaml`파일에 대한 가이드 문서입니다.



설정파일(`config.yaml`)이란

> 사용자가 지능컴포넌트를 정의하고자 할 때 규격이 되는 파일, 여러 개의 function을 정의한 yaml 파일

​    

## config.yaml 파일 구성

> Note

`config.yaml`파일의 구성은 다음과 같습니다. `list.yml` 에서 정의된 Runtime 버전의 기본값을 변경하고자 하는 경우, `build_args` 필드에 원하는 버전을 기입하면 됩니다.



```bash
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
​    

## build_args 필드 변경 확인

`build_args`필드에 버전변경을 한 것을 확인하려면 함수 컴포넌트를 배포시에 배포 로그를 확인하면 알 수 있습니다.



test라는 이름의 함수 컴포넌트에서 python3.5를 기본적으로 사용하기 위해 config.yaml파일을 다음과 같이 수정했다고 가정했을 때, 배포시 나타나는 로그에서 docker image의 버전을 확인할 수 있습니다.



**config.yaml**

```yaml
functions:
  hello-dcf:
    runtime: python
    desc: "This is test."
    maintainer: "KETI"
    handler:
      dir: ./test
      file: handler.py
      name: Handler
    image: keti.asuscomm.com:5001/test
    build_args:
    - PYTHON_VERSION=3.5
    build_packages:
      - make
      - python3-pip
      - gcc
      - python-numpy
dcf:
  gateway: keti.asuscomm.com:32222
```

​    

**deploy**

```bash
$ dcf function create -f config.yaml -v

>>
Building: test, Image:keti.asuscomm.com:5001/test
Sending build context to Docker daemon  4.096kB
Step 1/21 : ARG PYTHON_VERSION=3.4
Step 2/21 : FROM python:${PYTHON_VERSION}
3.5: Pulling from library/python
bc9ab73e5b14: Already exists
193a6306c92a: Already exists
e5c3f8c317dc: Already exists
a587a86c9dcb: Already exists
72744d0a318b: Already exists
6598fc9d11d1: Pull complete
74d2ee7772b2: Pull complete
ab2e66176e69: Pull complete
2c4175ee7cad: Pull complete
Digest: sha256:ef14a52ee8bacfa498b46ef1620ae3da16b0cbda8286b1f0a1a81aa71ac3a818
"Status: Downloaded newer image for python:3.5 -> 해당부분에서 build_args가 적용되었는지 확인"
```



​    

## 함수 컴포넌트 별 config.yaml 만들어 개별적으로 사용하기

앞서 설명한 가이드문서의 명령어를 이용하여 함수 컴포넌트를 배포할 경우, 모든 함수를 배포하게 되어 불필요한 작업이 생길 수 있습니다.



이를 조금 더 편리하게 작업하기 위해서 config.yaml을 함수별로 분리하거나, 하나의 config.yaml파일에서 함수를 따로 배포할 수 있는 기능에 대해서 알아보도록 하겠습니다.



### 1. 함수별 *.yaml 파일 생성하기

함수 컴포넌트를 생성할 시, 다음과 같은 명령어를 이용하면 다음과 함수 컴포넌트 별로 *.yaml 파일을 만들 수 있으며, 배포시에 해당 함수 컴포넌트만 배포할 수 있습니다.



**함수별 yaml 파일 생성**

```bash
$ dcf function init test --runtime python -f test.yaml

>>
.
├── dcf
├── dcf-runtime
├── test
└── test.yaml
```



**함수별 deploy 하기**

```bash
$ dcf function create -f test.yaml -v
```



​    

### 2. 단일의 config.yaml파일에서 특정한 함수 컴포넌트만 배포하기

아래와 같이 여러가지 함수 컴포넌트가 존재하고 여러가지 함수 컴포넌트에 대한 configuration 정보가 단일 config.yaml 에 명시되어있다고 가정하겠습니다.



```bash
.
├── AboutConfig_yaml.md
├── config.yaml
├── dcf
├── dcf-runtime
├── test1
├── test2
└── test3
```



이런 경우에서 특정 함수만 배포하고싶은 경우 다음과 같은 명령어를 이용하여 특정 함수만 배포할 수 있습니다.

```bash
$ dcf function create -f config.yaml test2
```

​        

##  config.yaml 규격

| Field       | Description                                                  | Example                                                      |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|             | 지능 컴포넌트의 이름                                         | echo-service                                                 |
| runtime     | 지능 컴포넌트가 실행될 환경                                  | python3                                                      |
| image       | 지능 컴포넌트 이미지 이름과 버전<br>(레포지토리인 keti.ausscomm.com:5001은 고정) | keti.asuscomm.com:5001/echo-service:v1                       |
| handler     | 지능 컴포넌트 배포시에 실행되는 엔트리포인트 정보            | handler:<br>&nbsp; name: Handler<br>&nbsp; dir: ./echo-service<br>&nbsp; file: "handler.py" |
| maintainer  | (optional)지능 컴포넌트 개발자 또는 유지보수 가능한 사람의 정보 | KETI                                                         |
| desc        | (optional)지능 컴포넌트 용도/설명                            | This is ....                                                 |
| environment | (optional)런타임 내에서 사용할 환경 변수                     | environment:<br>&nbsp; - "PATH=/usr/local/bin"               |
| skip_build  | (optional)지능 컴포넌트 빌드 및 레포지토리에 저장 단계 건너뛰기 | skip_build: true                                             |
| limits      | (optional)지능 컴포넌트가 사용할 자원 요청 및 제한           | limits:<br>&nbsp; cpu: "1"<br>&nbsp; gpu: "1"<br>&nbsp; memory: "1G" |
| build_args  | Dockerfile내에 ARG 값 지정                         | build_args:<br>&nbsp; - "PYTHON_VERSION=3.7"                 |
| build_packages  | Dockerfile내의 apt 패키지 관리자인 ADDITIONAL_PACKAGE 값으로 설정 (이 필드를 통해 원하는 패키지를 설치할 수 있습니다.)  | build_packages:<br>&nbsp; -make<br>&nbsp; -python3-pip<br>&nbsp; -gcc<br>&nbsp; -python-numpy                                                            |


