# Digital Companion Framework (DCF)

해당 문서는 디지털동반자 프레임워크에 대해서 설명하며, 이를 어떻게 사용하는지에 대한 가이드 문서입니다.

​    

## DCF Introduce



DCF는 FaaS(Function as a Service)의 구조를 따릅니다. 다양한 기관들은 DCF를 이용하여 규격화된 인공지능 모델을 배포할 수 있습니다. 아래 그림은 DCF의 구조에 대한 간략화된 설명입니다.



각 기관의 개발자들은 `DCF CLI` 를 이용하여 인공지능 모델을 규격화하고, 배포할 수 있습니다. 규격화된 인공지능 모델은 Docker기반으로 배포됩니다.



이렇게 규격화된 인공지능 모델은 오른쪽에 보이는 하나의 Function이 될 수 있으며, 일반 사용자(유저) 및 DCF를 이용해 상위 어플리케이션을 개발하려는 개발자들은 각 기관이 배포한 Function을 Call하는 것을 통해서 인공지능 모델의 추론 결과를 얻을 수 있습니다.



![DCF-concept](https://user-images.githubusercontent.com/13328380/47892857-590c2500-de9d-11e8-8989-7821892b1a72.png)



#### Reference

[1. Apache OpenWhisk - 소개 및 아키텍쳐](https://developer.ibm.com/kr/cloud/2017/12/24/apache-openwhisk-intro-architecture/)

[2. (번역) 서버리스 아키텍처](https://blog.aliencube.org/ko/2016/06/23/serverless-architectures/)



​    

## DCF Guide

**개발자**와 **사용자**에 따라 분류됩니다.


<br>

[개발자를 위한 가이드라인](https://github.com/DigitalCompanion-KETI/DCFramework/blob/feature/%2330/Developer/README.md) <br> : DCF 개발 환경 구축과 DCF 컴파일 과정을 설명합니다.

<br>

[사용자를 위한 가이드라인](https://github.com/DigitalCompanion-KETI/DCFramework/blob/feature/%2330/User/README.md) <br> : DCF CLI 설치와 DCF 사용법을 설명합니다.


