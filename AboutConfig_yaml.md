# About config.yaml



해당 문서는 함수 컴포넌트를 생성할 때, 만들어지는 `config.yaml`파일에 대한 가이드 문서입니다.



설정파일(`config.yaml`)이란

> 사용자가 지능컴포넌트를 정의하고자 할 때 규격이 되는 파일, 여러 개의 function을 정의한 yaml 파일

​    

## config.yaml 파일 구성

> Note

`config.yaml`파일의 구성은 다음과 같습니다. `list.yml` 에서 정의된 Runtime 버전의 기본값을 변경하고자 하는 경우, 

`build_args` 필드에 원하는 버전을 기입하면 됩니다.



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


