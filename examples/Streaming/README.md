## DCF Streaming

본 장에서는 Streaming  기능을 위한 통신 Architecture와 Protobuf 컴파일 방법을 소개한다. 



### Architecture

DCF 플랫폼의 스트리밍 통신 환경은 gRPC 를 통해 구현되었다. gRPC는 HTTP2 기반의 프로토콜로 개발 언어의 호환성과 빠른 속도로 마이크로 서비스간 통신에 권장되는 프로토콜이다. 현재 gRPC 에서 제공하는 통신 환경은 4가지가 있으며 다음과 같다.

![4way](https://user-images.githubusercontent.com/46108451/66361019-21b0cd00-e9b8-11e9-8ee8-c8302187e674.PNG)

- Unary RPC, 클라이언트가 서버로 Stub를 사용하여 통신 요청을 보내고 응답을 기다린다. 응답 처리 후 통신은 종료된다.
- Server Streaming RPC, 클라이언트가 서버로 통신 요청을 보내 스트림 포트 정보를 얻는다. 이 스트림 포트를 통해 서버에서 응답받는 스트리밍 데이터들을 처리한다.
- Client Streaming RPC, 클라이언트가 서버를 통해 스트리밍 데이터를 서버로 전송한다. 클라이언트가 메시지 작성이 끝나면 서버가 메시지를 읽고 응답을 보낼까지 기다린다.
- Bi-Directional Streaming RPC, read/write stream을 양방향으로 스트리밍 데이터를 보낼 수 있다. 두 개의 stream이 독립적으로 동작하기 때문에 서버/클라이언트는 순서에 관계없이 교대로 읽기/쓰기 처리가 가능하다.
  
  

현재 DCF 플랫폼 내 통신 환경은 Unary, Bi-Directional Streaming을 제공 중이며 Sever Streaming 을 구현 중이다.

### Streaming Prrotobuf

gRPC은 통신 구조를 정의하기 위한 Protobuf이 필요하며, 정의한 데이터로만 통신이 가능하다. 현재 DCF에서 정의된 Streaming Protobuf 의 통신 구조는 다음과 같다.

```protobuf
rpc Invokes(stream InvokeServiceRequest) returns (stream Messages) {}

message InvokeServiceRequest {
  string Service = 1;
  bytes Input = 2;
}

message Messages{
  bytes Output = 1;
}
```

발신, 수신 데이터 type 에 대해 bytes로 선언하였다. 이는 통신 데이터 호환을 위한 것으로 통신을 위한 byte array 변환이 필요하다.

##

또한, DCF 통신을 위한 gRPC Protobuf 정의가 필요하다. 다음의 명령을 통해 `Pb` 폴더의 `Gateway.proto` 을 컴파일한다.  컴파일 언어는 `python` 이다. 

```protobuf
python -m grpc_tools.protoc -I${GOPATH}/src/github.com/digitalcompanion-keti/pb \ 
            --python_out=. \
             --grpc_python_out=. \
            ${GOPATH}/src/github.com/digitalcompanion-keti/pb/gateway.proto
```

컴파일 후 실행 폴더 내 `gateway_pb2.py` 와 `gateway_pb2_gprc.py` 이 생성된다.



### Streaming Example

DCF 플랫폼의 Streaming 을 위한 예제를 소개한다. `python client` 와 `Hanlder.py `함수를 구현할 것이며 구성은 다음과 같다.



Bi-Directional Streaming RPC for Video 

[개발중] Server Streaming RPC with Stateful






