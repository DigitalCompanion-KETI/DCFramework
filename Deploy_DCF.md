# 3. Deploy DCF

DCF(Digital Companion Framework) 컴파일을 완료하였다면, 이제 DCF를 배포해보자

# Deploy

아래 명령어를 이용하여 **DCF**를 배포한다

    $ cd $GOPATH/scr/github.com/digitalcompanion-keti/dcf
    $ kubectl apply -f ./namespaces.yml
    $ kubectl apply -f ./yaml
    $ kubectl get pods --all-namespaces
    >>
    NAMESPACE     NAME                                   READY   STATUS    RESTARTS   AGE
    dcf           gateway-9b8b95cd8-5snlv                1/1     Running   0          36s
    dcf           gateway-9b8b95cd8-74c72                1/1     Running   0          36s
    dcf           prometheus-5c8f7f7c7d-zx5ks            1/1     Running   0          37s
    kube-system   coredns-fb8b8dccf-5k2xd                1/1     Running   13         4h10m
    kube-system   coredns-fb8b8dccf-snl9f                1/1     Running   12         4h10m
    kube-system   etcd-minikube                          1/1     Running   10         4h9m
    kube-system   kube-addon-manager-minikube            1/1     Running   10         4h9m
    kube-system   kube-apiserver-minikube                1/1     Running   10         4h9m
    kube-system   kube-controller-manager-minikube       1/1     Running   10         4h9m
    kube-system   kube-proxy-rl2wg                       1/1     Running   9          4h10m
    kube-system   kube-scheduler-minikube                1/1     Running   10         4h9m
    kube-system   kubernetes-dashboard-d7c9687c7-k7rzn   1/1     Running   13         4h10m
    kube-system   storage-provisioner                    1/1     Running   13         4h10m

- `STATUS`가 **Running**이 아닌 경우에는 [링크](https://kubernetes.io/ko/docs/reference/kubectl/cheatsheet/)를 참조하여 포드의 로그를 확인한다

# Verify Deploy

`dcf-cli`를 이용해 `echo`함수를 배포하여 `dcf`의 작동을 테스트한다

## Create folder for CLI testing

    $ mkdir cli-test
    $ cd cli-test

## Cloninig `dcf-runtime`

    $ git clone https://github.com/DigitalCompanion-KETI/dcf-runtime.git

## Create DCF function

- 함수를 배포하기 위해 함수의 initalization을 진행(runtime 설정, 함수 이름 설정 및 config.yaml 파일 생성)
- `<RUNTIME NAME> `은 `go`  ,`python`을 지원한다 

```bash
$ dcf-cli function init <FUNCTION NAME> --runtime <RUNTIME NAME> 

>> 
Folder: <FUNCTION NAME> created
Fucntion handler created in folder: <FUNCTION NAME>/src
Rewrite the function handler code in <FUNCTION NAME>/src folder
Config file written: config.yaml

$ cd <FUNCTION NAME>
```

## Configure `config.yaml`

* `config.yaml` 에서  `gpu`  값으로 사용할 gpu의 사용 개수를 정할 수 있다. 

* `gpu : ""`  으로 옵션을 줄 경우,  cpu로 해당 함수가 할당된다.
  
      $ cd config.yaml
      >>
      functions:
        echo:
          runtime: go
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
      dcf:
        gateway: keti.asuscomm.com:32222

## Building Function

- Kubernetes에 생성한 함수를 배포하기 위한 도커 이미지 생성 
  
  ```
  $ dcf-cli function build -f config.yaml
  
  >> 
  Building function (echo) ...
  
  Image: keti.asuscomm.com:5001/echo built in local, successfully.
  ```
* `Python` 의 경우, 도커 라이브러리 연동문제로 에러가 발생한다.  따라서 해당 함수 폴더의 `requirements.txt` 에 사용할 라이브러리를 명시하고 함수 빌드를 실행한다.
  
  ```
  $ ls
   >>
   config.yaml dcf-runtime Dockerfile handler.py requirements.txt src
  #사용할 python lib을 명시한다.
  $ vi requirments.txt
  numpy
  pandas
  ```

## Deploying Funtion

- 생성된 이미지를 통해 Kubernetes에 함수 배포
  
  ```bash
  $ dcf-cli function deploy -f config.yaml
  
  >> 
  Is docker registry(registry: keti.asuscomm.com:5001) correct ? [y/n] y
  Pushing: echo, Image: keti.asuscomm.com:5001/echo in Registry: keti.asuscomm.com:5001 ...
  Deploying: echo ...
  Deploying DCF function, successfully
  ```

## Confirm DCF function list

- Kubernetes에 배포가 완료된 함수의 목록 확인한다. 서버에 배포가 완료되기까지 수 초가 소요되며  `Status`  값을 통해 확인한다. `Ready ` 상태가 되면 정상적으로 배포가 완료된 것이다.
  
  ```bash
    $ dcf-cli function list
  
    >> 
    Function    Image           Maintainer    Invocations    Replicas    Status    Description
    echo        $(repo)/echo                  0              1           Ready
  ```

## Verify deployed function using invoke

- Kubernetes에 배포된 함수를 호출
  
  ```bash
    $ echo "Hello" | dcf-cli function call echo
  
    >> 
    Hello
  ```

### REFERENCE

1. [Create a Secret based on existing Docker credentials](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/)
