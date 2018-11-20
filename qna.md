# Q&A

해당 문서는 DCF를 사용하면서 겪을 수 있는 이슈들에 대한 대표적인 Q&A를 모아놓은 문서입니다.

​    

#### 1. 함수 컴포넌트의 GPU 사용 여부

현재 DCF 함수 컴포넌트들에 대해서 GPU지원을 하고있지 않습니다. 딥러닝 모델을 사용할 경우 CPU버전의 패키지 혹은CPU에서 작동하는 코드 셋팅을 부탁드리겠습니다.

​    

#### 2. 함수 컴포넌트에서의 Train 여부

Train은 지원하지 않습니다. 배포하려는 모델을 미리 학습시켜 weights파일을 확보한 후에, 배포해주시면 되겠습니다.

​    

#### 3. 전역변수 및 STATE 저장 여부

DCF는 FaaS기반의 프레임워크이므로, invoke(함수 컴포넌트 호출)이 일어난 후에는 함수 스코프의 변수 및 인스턴스들이 없어지게 됩니다. 따라서 특정 STATE를 갖는 시퀸스 데이터를 봐야하는 모델은 아직 DCF에서 사용 불가능합니다.



weigths파일도 초기에만 load 후, 지속적으로 load된 weigths를 사용하는 방식이 아니므로, weights를 지속적으로 load해주셔야하며, 변수 및 인스턴스가 비동기식으로 자원 해제가 되므로, 꼭 weights sharing option을 체크해주셔야합니다. (Tensorflow 같은 경우에는 tf.AUTO_REUSE 옵션을 꼭 필수로 선택해주셔야합니다.)


- 해당 내용에 대한 지원 여부는 내부적으로 논의 중입니다.


​    
#### 4. 컴파일을 해야지만 사용가능한 패키지의 경우는 어떻게 처리할 수 있는지 

현재 DCF는 requirements.txt로 모든 것을 설치할 수 있다면, 개발자는 간단하게 딥러닝 모델을 올리고, 배포할 수 있는 구조를 가지고 있으나, 직접 컴파일을 해서 사용해야만 하는 패키지의 경우에는 Dockerfile을 수정해주셔야합니다. 해당 내용에 대해서는 [튜토리얼 문서(SSD(Object Detection) Component](../SSD(Object_Detection)_Component_Tutorial.md))에서 제공하고 있으니 이를 참고하시면 됩니다.

​    

#### 5. Docker Image를 직접 DCF에 전달할 수는 없는지?

현재 DCF는 Docker Image를 직접 전달해서 배포할 수 있게 지원하고 있지 않습니다.


#### 6. 사내 내부망의 방화벽으로 인한 `dcf-cli` timeout 발생시, 방화벽 해제대상 IP & PORT
IP : keti1.asuscomm.com 혹은 keti.asuscomm.com
PORT : 5001

을 대상으로 방화벽을 오픈하면, `dcf-cli`를 이용하여 DCF에 접속할 수 있습니다.
