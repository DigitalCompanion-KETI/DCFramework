# gRPC Guide 

이 가이드는 gRPC 설치법 및 사용법에 대한 가이드이다. 이에 앞서, RPC란 네트워크 상 원격에 있는
 서버의 서비스를 호출하는데 사용되는 프로토콜로 IDL(Interface Definition Language)로 인터페이스를 정의한 후 이에 해당하는 Skeleton과 Stub 코드, 즉 해당 프로그래밍 언어가 부를 수 있는 형태의 코드를 통해 프로그래밍 언어에서 호출해서 사용하는 방식이다. gRPC란 자바, C/C++ 뿐만 아니
라 Python, Ruby, Go 등 다양한 언어들을 지원함으로써 서버 간 뿐만 아니라 클라이언트 어플리케이
션이나 모바일 앱에서도 사용 가능한 RPC 프레임워크이다.

## gRPC for Go 

### PREREQUISITES 

- Go 1.7 이상의 버전이 필요하다.
- [GOPATH](https://golang.org/doc/code.html#GOPATH)가 설정되어야 한다.

### Install gRPC 

```
$ go get -u google.golang.org/grpc
$ go get -u golang.org/x/net
$ go get -u golang.org/x/sys/unix
```

### Install Protocol Buffers 

- [here](https://github.com/google/protobuf/releases)에서 버전과 플랫폼에 맞는 압축 파일의 링
크를 복사하여 파일을 다운 받는다.

```
$ wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-linux-x86_64.zip
$ unzip protoc-3.6.1-linux-x86_64.zip -d protoc3
$ sudo mv protoc3/bin/* /usr/local/bin/
$ export PATH=$PATH:/usr/local/bin
$ sudo mv protoc3/include/* /usr/local/include/
```

### Install Protoc plugin 

```
$ go get -u github.com/golang/protobuf/{proto,protoc-gen-go}
$ export PATH=$PATH:$GOPATH/bin
```

## gRPC 통신 

### 1. Define gRPC service 
디지털 동반자의 지능 컴포넌트를 호출하기 위해 다음과 같은 gRPC service 정의가 필요하다.

  - 파일의 확장자는 [.proto]이고 Service의 이름은 proto 파일의 이름과 같다.

```
$ mkdir -p ${GOPATH}/src/pb
$ vim ${GOPATH}/src/pb/gateway.proto
```

```
syntax = "proto3";

package pb;

service Gateway {
  rpc Invoke(InvokeServiceRequest) returns(Message) {}
}

message Message {
  string Msg = 1;
}

message InvokeServiceRequest {
  string Service = 1;
  bytes Input = 2;
}
```

### 2. Generate gRPC service 

``` 
$ protoc -I . \
  -I${GOPATH}/src/pb \
  --go_out=plugins=grpc:${GOPATH}/src/pb \
  ${GOPATH}/src/pb/gateway.proto
```

- 컴파일을 완료하면 컴파일할 때의 설정값인 ${GOPATH}/src/pb 경로에 gateway.pb.go 파일이 생성된다.

### 3. Create gRPC client 
다음은 디지털 동반자의 특정 지능 컴포넌트를 호출하는 메소드인 "Invoke"를 사용하는 예제이다. 입력 인자로 사용될 InvokeServiceRequest에 DCF CLI로 생성한 지능 컴포넌트 이름(Service)과 지능 컴포넌트에 전달할 입력 값(Input)을 넣어 Invoke를 호출한다.

> ### For Beginning

- 구현된 서비스 메소드를 호출하기 위해, 서버와 통신할 수 있는 gRPC 채널을 만든다

- 채널이 구축되면 RPC를 수행할 클라이언트 Stub을 만든다

- 서비스 메소드를 호출한다

```go
package main

import (
    "context"
    "fmt"
    "log"

    "pb"
    "google.golang.org/grpc"
)

func main() {
    // Creating a gRPC Channel
    address := "keti.asuscomm.com:32222"

    conn, err := grpc.Dial(address, grpc.WithInsecure())
    if err != nil {
      log.Fatalf("did not connect: %v", err)
    }

    defer conn.Close()
    
    ctx := context.Background()

    // Creating a stub
    client := pb.NewGatewayClient(conn)

    // Calling service method
    r, err := client.Invoke(ctx, &pb.InvokeServiceRequest{Service: "echo-service", Input: []byte("hello world")})
    if err != nil {
        log.Fatalf("could not invoke: %v\n", err)
    }

    fmt.Println(r.Msg)
}
```

## gRPC for Python 

### PREREQUISITES 

- ### Install pip (version 9.0.1 ~)

  gRPC Python은 Python 2.7 또는 Python 3.4 이상부터 지원 가능하다.

  ```
  $ python -m pip install --upgrade pip
  ```

  만약 root 계정이 아닌 경우, 다음과 같이 pip을 업그레이드 할 수 있다.

  ```
  $ python -m pip install virtualenv
  $ virtualenv venv
  $ source venv/bin/activate
  (venv) $ python -m pip install --upgrade pip
  ```

### Install gRPC 

```
$ python -m pip install grpcio
```

* OS X EL Capitan의 운영체제일 경우, 다음과 같은 오류가 발생한다.

  $ OSError: [Errno 1] Operation not permitted: '/tmp/pip-qwTLbI-uninstall/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/six-1.4.1-py2.7.egg-info'

* 이 오류는 다음과 같이 해결할 수 있다.

```
$ python -m pip install grpcio --ignore-installed
```

### Install gRPC tools 

* Python gRPC tool은 프로토콜 버퍼 컴파일러인 protoc와 proto 파일로부터 서버 / 클라이언트 코드를 생성하는 특별한 플러그인이 포함되어있다. 때문에 Golang gRPC에서처럼 따로 프로토콜 버퍼를
 설치할 필요가 없다.

```
$ python -m pip install grpcio-tools googleapis-common-protos
```

## gRPC 통신 

### 1. Define gRPC service 
디지털 동반자의 지능 컴포넌트를 호출하기 위해 다음과 같은 gRPC service 정의가 필요하다.

```
$ vim gateway.proto
```

```
syntax = "proto3";

package pb;

service Gateway {
  rpc Invoke(InvokeServiceRequest) returns(Message) {}
}

message Message {
  string Msg = 1;
}

message InvokeServiceRequest {
  string Service = 1;
  bytes Input = 2;
}
```

### 2. Generate gRPC service 

```
$ python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. gateway.proto
```

### 3. Create gRPC client 
다음은 디지털 동반자의 특정 지능 컴포넌트를 호출하는 메소드인 "Invoke"를 사용하는 예제이다. 입력 인자로 사용될 InvokeServiceRequest에 DCF CLI로 생성한 지능 컴포넌트 이름(Service)과 지능 컴포넌트에 전달할 입력 값(Input)을 넣어 Invoke를 호출한다.

> ### For Beginning
- 구현된 서비스 메소드를 호출하기 위해, 서버와 통신할 수 있는 gRPC 채널을 만든다

- 서비스 메소드를 호출하기 위한 Stub을 만든다

- 서비스 메소드를 호출한다.

```python
import grpc

import gateway_pb2
import gateway_pb2_grpc

def run():
    # Creating a channel
    channel = grpc.insecure_channel('keti.asuscomm.com:32222')
    # Creating a stub
    stub = gateway_pb2_grpc.GatewayStub(channel)

    # Calling service methods
    servicerequest = gateway_pb2.InvokeServiceRequest(Service="echo-service", Input="hello world".encode())
    r = stub.Invoke(servicerequest)
    print(r.Msg)

if __name__ == '__main__':
    run()
```
