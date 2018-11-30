
# Text기반 단발성 대화문의 다점주 감정 인식 API

## 개요
 - 분석 대상 문장에 대하여 7가지 다범주 감정으로의 분류 확률 산출과 가장 높은 확률을 가진 감정을 해당 문장의 감정으로 출력

![image](https://user-images.githubusercontent.com/18477915/49271091-95886c00-f4af-11e8-8929-09798dd82096.png)

## 활용 예
 1. 요청 파라미터{ “text“ : “분석할 문장“}  
   
    text : String타입, 필수, 분석할 문장 데이터
 
 
 2. 출력 결과
 
    (Happiness-10001, Anger-10002, Disgust-10003, Fear-10004, Neutral-10005, Sadness-10006, Surprise-10007)  
 
 
 3. 출력 결과 예시
 
         {“emotion_per“:         
            {“10001“: “해당 감정 확률” 
             “10002“: “해당 감정 확률” ,
             “10003“: “해당 감정 확률” ,
             “10004“: “해당 감정 확률” ,         
             “10005“: “해당 감정 확률” ,
             “10006“: “해당 감정 확률” ,
             “10007“: “해당 감정 확률”,}     
            predict_emotion“ : “예측 감정”}
     
     
     10001 ~ 10007 : String타입, 필수, 각 감정에 대한 모델의 예측 확률  
     
     predict_emotion : String타입, 필수, 입력 문장에 대한 예측 감정
   
   ## API 접근 경로
   ### https://ibis_api.hanynag.ac.kr:5000/sentiment_analysis?text= "분석할문장"
