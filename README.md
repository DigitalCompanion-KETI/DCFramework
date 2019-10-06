# DCF - Digital Companion Framework

<p align="center">
    <img src="https://user-images.githubusercontent.com/13328380/66203965-9e0b8d80-e6e4-11e9-948d-9faa71a5d97c.png?style=centerme"/>
</p>

---

디지털 동반자 프레임워크는 인공지능 모델을 배포하기 위한 서버리스 프레임워크 입니다. 

#### Highlights

- 엔비디아 도커(Nvidia-Docker)를 이용한 인공지능 모델 패키징 지원
- RESTful / gRPC 프로토콜 지원
- 스트리밍 구조 지원
- 함수 실행 테스트 지원
- CLI를 이용한 쉬운 함수 배포
- Auto-Scale 지원

## Overview of Digital Companion Framework

### Architecture

![Architecture](https://user-images.githubusercontent.com/13328380/66216078-c1900180-e6ff-11e9-943b-463c55ddec3b.png)



##### Gateway

게이트웨이는 HTTP, gRPC 요청을 받아 모두 gRPC 요청으로 변경하며 요청에 따라 여러 작업을 수행한다. 주로 노드들 사이에 배포된 여러 함수들을 찾아서 호출하고 결과값을 반환해주는 역활을 하며 각 함수의 호출 수를 카운트한다.



##### Watcher

와처는 노드에 배포되어있는 함수 그 자체이다. 와처는 사용자가 작성한 외부 함수를 로드(load)하여 구동되며 사용자 요청을 기다린다. 사용자의 요청이 게이트웨이로 들어와서 와처가 호출되면 와처는 외부 함수를 호출하여 결과값을 반환한다.



### Runtime

디지털 동반자 프레임워크는 함수의 런타임으로 아래와 같은 언어를 지원하며 함수를 생성 시 런타임을 명시한다.

- Golang

- Python3.6



**Golang example**

```bash
$ dcf-cli function init echo --runtime go
```

**Python example**

```bash
$ dcf-cli function init echo --runtime python
```



### Supported GPU

클러스터 각 노드에 엔비디아 그래픽 드라이버가 설치되어있다면 디지털 동반자 프레임워크는 GPU를 함수에서 사용할 수 있게 지원한다. 

> 전자부품연구원 휴먼IT센터의 클러스터 환경을 이용하는 경우 GPU 사용량에 대해서 전자부품연구원과 협의 후에 사용해야한다.



#### GPU configuration of Function

디지털 동반자 프레임워크에서 함수를 생성하면 만들어지는 **config.yaml**에서 GPU 관련 옵션을 변경하여 GPU 자원을 할당받을 수 있다.

```yaml
functions:
  echo:
    runtime: python
    desc: ""
    maintainer: ""
    handler:
      dir: ./src
      file: ""
      name: Handler
    docker_registry: keti.asuscomm.com:5001
    image: keti.asuscomm.com:5001/echo
    limits:
      memory: ""
      cpu: ""
      gpu: ""
    build_args:
    - CUDA_VERSION=9.0
    - CUDNN_VERSION=7.4.1.5
    - UBUNTU_VERSION=16.04
dcf:
  gateway: keti.asuscomm.com:32222
```

**Option of GPU**

| Option                   | Description |
| ------------------------ | ----------- |
| limits.gpu               | 지원하는 GPU 개수 |
| build_args.CUDNN_VERSION | CUDA 버전     |
| build_args.CUDA_VERSION  | CuDNN 버전    |



## Get Started

- CLI installation

- Usage DCF-CLI

  - runtime list

  - Create function

  - Write handler

  - Build function

  - Test function

  - Deploy function

  - Invoke function

  - Log of function

  - change gateway address

- Python package(dcfgrpc)

- More Examples

- Installation DCF in own environment
