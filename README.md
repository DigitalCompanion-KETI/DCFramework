# Digital Companion Framework (DCF)

해당 문서는 디지털동반자 프레임워크에 대해서 설명하며, 이를 어떻게 사용하는지에 대한 가이드 문서입니다.

​    

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

  사용하기 위해서 DCF CLI가 이미 설치되어 있다면, 기존의 설치된 DCF CLI를 삭제하고 새로 받아주시기 바랍니다.

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

[4. config.yaml 파일 구성](AboutConfig_yaml.md)

[5. gRPC Guide](grpc-guide.md)

[6. Q&A](qna.md)




