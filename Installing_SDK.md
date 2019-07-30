# 1. Installing SDK

 DCF(Digital Companion Framework) 개발 환경을 위한 VM, Minikube, kubectl,  Docker을 설치한다.

## Installing VM

미니쿠베를 시작하기 전, [버츄얼박스](https://www.virtualbox.org/)를 설치하여 미니쿠베가 사용할 수 있는 가상머신을 설치한다.  사용하고 있는 OS가 Mac과 리눅스 환경이라면,  가상머신 설치를 생략한다.

# Installing Minikube

미니쿠베(Minikube)는 쿠버네티스(Kubernetes)처럼 클러스터를 구성하지 않고 단일 컴퓨팅환경(노트북, 데스크탑 등)에서 쿠버네티스 환경을 만들어준다.

본 가이드에선 MacOs와 Ubuntu18.04를 기준으로 테스트하여 작성하였다.

## MacOS

    $ brew cask install minikube

## Ubuntu

    $ curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64 \
      && chmod +x minikube
    $ sudo cp minikube /usr/local/bin && rm minikube

# Start Minikube

가상머신 미설치 사용자의 경우, 아래와 같은 명령어로 미니쿠베를 시작한다

    $ sudo minikube start --insecure-registry="<IP ADDRESS>:<PORT>"

- `<IP ADDRESS>:<PORT>` : 앞으로 생성할 도커 레지스트리 서버의 주소와 포트번호를 적어준다. 본 가이드에서는 호스트 OS(VM)의 IP와 포트번호,  5000으로 작성하였다
- 가상머신에서 미니쿠베를 사용하는 경우, `--vm-driver` 옵션을 `none`으로 설정하여 시작한다
  
        $ echo export CHANGE_MINIKUBE_NONE_USER=true >> ~/.bashrc
        $ sudo minikube start --vm-driver=none --insecure-registry="<IP ADDRESS>:<PORT>"

`~/.kube`,  `~/.minikube` 폴더의 권한을 `$USER`로 변경한다

    $ sudo chown -R $USER ~/.kube ~/.minikube

# Installing kubectl

`kubectl`은 쿠버네티스를 제어하기 위한 명령 줄 인터페이스이다

아래와 같은 명령어로 kubectl을 설치한다

## MacOS

    $ brew install kubernetes-cli
    $ kubectl version

## Ubuntu

    $ curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl
    $ curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.14.0/bin/linux/amd64/kubectl
    $ chmod +x ./kubectl
    $ sudo mv ./kubectl /usr/local/bin/kubectl
    $ kubectl version

# Verify installed minikube

    $ kubectl get pods --all-namespaces
    >>
    NAMESPACE     NAME                               READY   STATUS             RESTARTS   AGE
    kube-system   coredns-fb8b8dccf-4bq7x            1/1     Running   0          113s
    kube-system   coredns-fb8b8dccf-jw6j2            1/1     Running   0          113s
    kube-system   etcd-minikube                      1/1     Running   0          4m19s
    kube-system   kube-addon-manager-minikube        1/1     Running   0          4m22s
    kube-system   kube-apiserver-minikube            1/1     Running   0          4m17s
    kube-system   kube-controller-manager-minikube   1/1     Running   0          4m6s
    kube-system   kube-proxy-h8q7p                   1/1     Running   0          5m11s
    kube-system   kube-scheduler-minikube            1/1     Running   0          4m16s
    kube-system   storage-provisioner                1/1     Running   0          5m7ss

## CrashLoopBackOff Error

버츄얼박스에서 미니쿠베를 실행할 경우 아래와 같이 에러가 발생할 수 있다.

    $ kubectl get pods --all-namespaces
    >>
    NAMESPACE     NAME                               READY   STATUS             RESTARTS   AGE
    kube-system   coredns-fb8b8dccf-mtn7d            0/1     CrashLoopBackOff   5          3m54s
    kube-system   coredns-fb8b8dccf-t584j            0/1     CrashLoopBackOff   5          3m54s
    kube-system   etcd-minikube                      1/1     Running            0          2m46s
    kube-system   kube-addon-manager-minikube        1/1     Running            0          4m1s
    kube-system   kube-apiserver-minikube            1/1     Running            0          2m51s
    kube-system   kube-controller-manager-minikube   1/1     Running            0          2m52s
    kube-system   kube-proxy-rtswf                   1/1     Running            0          3m54s
    kube-system   kube-scheduler-minikube            1/1     Running            0          2m51s
    kube-system   storage-provisioner                1/1     Running            0          3m53s

아래 명령어로 이를 해결할 수 있다.

**CoreDns configmap** 수정한다. 아래 명령어를 실행 후, `loop`라는 단어를 삭제한다

    $ kubectl -n kube-system edit configmap coredns

기존 포드를 삭제하고 새로운 설정이 적용된 포드를 생성한다

    kubectl -n kube-system delete pod -l k8s-app=kube-dns

## Installing Docker

### Docker-CE

* 우분투 18.04 기준으로 테스트하였다

```
$ sudo apt-get remove docker docker-engine docker.io containerd runc
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
$ docker --version

$ sudo usermod -aG docker $USER # reboot after executing
```

### htpasswd

```
$ sudo apt-get install apache2-utils
```

### Start docker registry

`Insecure registry`란 docker image를 push & pull 하기 위한 개인 저장소(private registry)이다.  사용자가 build한 docker image를 dcf 저장소에 저장하기 위해 다음과 같이 설정을 해주고 docker를 재시작한다.

```
$ echo '{"insecure-registries": ["keti.asuscomm.com:5001"]}'>> /etc/docker/daemon.json
$ service docker restart
```

Docker 재시작 후, 다음과 같은 명령어로 설정을 확인한다.

```
$ sudo docker info
>>
Insecure Registries:
 keti.asuscomm.com:5001
```

### Check Login to docker registry

도커 레지스트리에 로그인이 잘 되는지 확인한다

```
$ docker login keti.asuscomm.com:5001
Username: elwlxjfehdqkswk
Password: elwlxjfehdqkswk
```

### 

### REFERENCE

1. [Minikube 설치](https://kubernetes.io/ko/docs/tasks/tools/install-minikube/)
2. [coredns pods have CrashLoopBackOff or Error state](https://stackoverflow.com/a/53414041/2153777)
3. [Sharing a local registry with minikube](https://blog.hasura.io/sharing-a-local-registry-for-minikube-37c7240d0615/)
