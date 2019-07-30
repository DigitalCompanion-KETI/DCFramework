# 2. Compile DCF

DCF(Digital Companion Framework)를 컴파일한다. 컴파일 이전에 **미니쿠베**와 **도커 레지스트리** 구축이 완료되어있어야 한다. 또한 1장에서 설명한 도커 로그인이 필요하다

# Requirements

### Go

[공식 Go 다운로드 홈페이지](https://golang.org/doc/install)에서 설치파일을 다운로드 받는다

아래 명령어를 이용하여 설치파일을 압축해제하고, `/usr/local`로 위치를 옮긴다. (*우분투 18.04 / Go 1.12.5 버전에서 확인했다* )

    $ sudo tar -xvf go1.12.5.linux-amd64.tar.gz
    $ sudo mv go /usr/local

`.bashrc` 파일을 수정하여 go와 관련된 환경변수를 설정한다.

    $ vim ~/.bashrc
    >>
    # add this lines
    export GOROOT=/usr/local/go
    export GOPATH=$HOME/go
    export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

변경한 환경변수 파일을 적용한다

    $ source ~/.bashrc

Go 설치를 확인한다

    $ go version
    $ go env

### gRPC

    $ go get -u google.golang.org/grpc
    # $ go get -u golang.org/x/net
    $ go get -u golang.org/x/sys/unix

### gRPC-gateway

    $ go get -u github.com/kardianos/govendor
    $ cd $GOPATH/src
    $ govendor init
    $ govendor fetch github.com/googleapis/googleapis/google/api
    $ cd $GOPATH/src/github.com/golang/protobuf
    $ git checkout master
    $ go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway
    $ go get -u github.com/grpc-ecosystem/grpc-gateway/protoc-gen-swagger
    $ go get -u github.com/golang/protobuf/protoc-gen-go

### protocol buffers

    $ curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/protoc-3.7.1-linux-x86_64.zip
    $ unzip protoc-3.7.1-linux-x86_64.zip -d protoc3
    
    $ sudo mv protoc3/bin/* /usr/local/bin/
    $ sudo mv protoc3/include/* /usr/local/include/
    
    $ sudo chown $USER /usr/local/bin/protoc
    $ sudo chown -R $USER /usr/local/include/google
    
    $ export PATH=$PATH:/usr/local/bin

### Protoc-gen-go

    $ cd $GOPATH/src/github.com/golang/protobuf/protoc-gen-go
    $ git checkout tags/v1.2.0 -b v1.2.0
    $ go install

### gRPC tools

    $ pip install grpcio-tools
    $ pip3 install grpcio-tools

### protoc plugin

    # $ go get -u github.com/golang/protobuf/{proto,protoc-gen-go}
    # $ export PATH=$PATH:$GOPATH/bin

### dep

    $ go get -u github.com/golang/dep/cmd/dep

# Compile DCF

먼저 아래와 같이 DCF 설치 파일을 생성하고 이동한다. 

    $ mkdir $GOPATH/src/github.com/digitalcompanion-keti
    $ cd $GOPATH/src/github.com/digitalcompanion-keti

## dcf

 `dcf` 와 `pb` 저장소를 `dcf` 의 폴더로 클론한다

    $ git clone https://github.com/DigitalCompanion-KETI/dcf.git dcf
    
    $ git clone https://github.com/DigitalCompanion-KETI/pb.git 

## dcf-gateway

`dcf-gateway`를 클론한다

    $ git clone https://github.com/DigitalCompanion-KETI/dcf-gateway.git
    $ cd dcf-gateway

`make` 명령어를 이용해서 `dcf-gateway`를 컴파일한다

    $ make build
    $ make push

## watcher

`dcf-watcher`를 클론한다

    # move path digitalcompanion-keti/
    $ cd ..
    
    $ git clone https://github.com/DigitalCompanion-KETI/dcf-watcher.git
    $ cd dcf-watcher

`make` 명령을 실행한다

    $ make

## cli

`dcf-cli`를 클론하고 의존성 패키지 설치 한다.

    $ git clone https://github.com/DigitalCompanion-KETI/dcf-cli.git
    $ cd dcf-cli
    $ dep init
    $ dep ensure

`make`명령을 실행한다

    $ make

`$GOPATH/bin`을 확인해보면 `dcf-cli`가 컴파일 되어있는 것을 확인할 수 있다

# Verify

컴파일 완료 후, `docker images`와 레지스트리에 있는 이미지를 확인했을 때, 아래와 같이 결과나 나오면 성공적으로 컴파일이 완료된 것이다.

    $ docker images
    >>
    REPOSITORY                                TAG                    IMAGE ID            CREATED             SIZE
    keti.asuscomm.com:5001/watcher            0.1.0-tensorflow_gpu   31f40b062eed        4 minutes ago       3.18GB
    keti.asuscomm.com:5001/watcher            0.1.0-tensorflow       c2c3e06920da        4 minutes ago       902MB
    keti.asuscomm.com:5001/watcher            0.1.0-python3          eb0f989c2d48        4 minutes ago       413MB
    keti.asuscomm.com:5001/watcher            0.1.0-python2          5ddd69549253        4 minutes ago       402MB
    keti.asuscomm.com:5001/watcher            0.1.0-go               106df11e2561        4 minutes ago       783MB
    keti.asuscomm.com:5001/watcher            <none>                 0083684bb780        About an hour ago   3.18GB
    keti.asuscomm.com:5001/watcher            <none>                 360eb8bb6a16        About an hour ago   902MB
    keti.asuscomm.com:5001/watcher            <none>                 99604a808f93        About an hour ago   413MB
    keti.asuscomm.com:5001/watcher            <none>                 86155829dd8f        About an hour ago   402MB
    keti.asuscomm.com:5001/watcher            <none>                 7d05557bbe67        About an hour ago   783MB
    keti.asuscomm.com:5001/gateway            0.1.0                  e7b626e81453        2 hours ago         95.3MB
    <none>                                    <none>                 b08f32dba190        2 hours ago         903MB
    <none>                                    <none>                 d902b7552ee4        2 hours ago         95.3MB
    <none>                                    <none>                 7349367e11ae        2 hours ago         903MB
    <none>                                    <none>                 3f9aebc32156        2 hours ago         95.3MB
    <none>                                    <none>                 bc6150068312        2 hours ago         903MB
