## Install DCF for user

사용자를 위한 가이드라인으로 DCF CLI 설치와 DCF 사용법을 설명합니다.  여기서 DCF를 설치한다는 의미는, DCF CLI를 설치한다는 의미가 됩니다. 

우리는 DCF CLI를 통해서 모든것(인공지능 모델 규격화, 배포)를 진행하게됩니다.

​    

### 1. Docker 설치

DCF CLI를 설치하기 전에, 먼저 해당 컴퓨터에 Docker가 설치되어있어야합니다.

도커란 컨테이너 기반의 오픈소스 가상화 플랫폼입니다. 다양한 프로그램, 실행환경을 

컨테이너로 추상화하고 동일한 인터페이스를 제공하여 프로그램의 배포 및 관리를 

단순하게 해주기 위해 도커를 사용합니다. 

도커 버전은 17.05-CE버전 이상을 요구합니다.

설치 방법은 [링크](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)의 `Install Docker CE`를 참조하시면 됩니다.

- docker >= 17.05-ce

​    

### 2. DCF CLI 다운로드

다음과 같은 명령어를 통해 DCF CLI를 다운받고, 해당 파일에 실행 권한을 설정합니다.

> Notify

- 지능 컴포넌트 생성 시, 자동으로 생성되는 config.yaml에서 기존에 없던 build_args와 build_packages 필드를 

  사용하기 위해서 DCF CLI가 이미 설치되어 있다면, 기존의 DCF CLI를 삭제하고 새로 받아주시기 바랍니다.

```bash
$ wget https://github.com/DigitalCompanion-KETI/DCFramework/releases/download/v0.1.0/dcf
$ chmod +x dcf
$ mv dcf /usr/local/bin
```

​    

### 3. DCF CLI 설정

DCF 저장소에 로그인하기 위해서 다음과 같이 설정을 해줍니다.

#### 3.1 Insecure registry

Insecure registry란 docker image를 push & pull 하기 위한 개인 저장소(private registry)입니다. 

사용자가 build한 docker image를 dcf 저장소에 저장하기 위해 다음과 같이 설정을 해주고 docker를 재시작합니다. 

```bash
$ echo '{"insecure-registries": ["keti.asuscomm.com:5001"]}'>> /etc/docker/daemon.json
$ service docker restart
```

- Docker 재시작 후, 다음과 같은 명령어로 `Insecure Registries`에 `keti.asuscomm.com:5001`이 들어가있는지를 통해 

  설정완료가 잘 되었음을 확인할 수 있습니다.

```bash
$ sudo docker info
>>
Insecure Registries:
 keti.asuscomm.com:5001
```

#### 3.2 Docker DCF Repository Login

Insecure registry가 dcf 저장소로 설정 완료 되었으면, dcf 저장소에 저장된 image를 pull 또는 

dcf 저장소에 image를 push 하기 위해서 다음과 같은 Username과 Password를 통해 login을 진행합니다.

```bash
$ docker login keti.asuscomm.com:5001
Username: elwlxjfehdqkswk
Password: elwlxjfehdqkswk
```


​    

## Tutorial

DCF설치를 완료했다면, 다음과 같은 가이드 문서를 통해서 DCF사용법에 대해서 확인할 수 있습니다.



[1. Hello DCF](helloDCF.md)

[2. Variety input data format](Variety_input_data_format.md)

[3. SSD(Object Detection) Component](SSD(Object_Detection)_Component_Tutorial.md)

[4. Python Client Example](Python_Client_Example.md)

[5. config.yaml 파일 구성](AboutConfig_yaml.md)

[6. gRPC Guide](grpc-guide.md)

[7. Q&A](qna.md)




