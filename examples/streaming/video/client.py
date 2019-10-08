import queue
import time
import datetime 
import threading

import argparse 
import numpy as np 
import cv2 

import grpc
import gateway_pb2
import gateway_pb2_grpc


address = '10.0.0.91'
port = 32222

class Client:
    def __init__(self):
        # create a gRPC channel 
        print("Start DCF Client for video Streaming..")
        channel = grpc.insecure_channel(address + ':' + str(port))
        self.conn = gateway_pb2_grpc.GatewayStub(channel)
        # create new listening thread for when new message streams come in
        self.dataQueue = queue.Queue()

        self.cap = cv2.VideoCapture(args.video)  
        # 화면 조절 
        self.cap.set(3, 960) 
        self.cap.set(4, 640) 

        threading.Thread(target=self.__listen_for_messages).start()
        self.Capture()

    def generator(self):
        """
        클래스 내 큐를 관리합니다. 
        """
        while True:
            time.sleep(0.01)
            if self.dataQueue.qsize()>0:
                yield self.dataQueue.get()

    def __listen_for_messages(self):
        """
        이 함수는 gRPC Streaming 의 수신 메세지를 처리합니다. 
        """
        time.sleep(5)
        responses = self.conn.Invokes(self.generator())

        try :
            for i in responses:
                nparr = np.frombuffer(i.Output, np.uint8)
                newFrame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow("DCF Streaming", newFrame)
                k = cv2.waitKey(1) & 0xff 
                if k == 27: # ESC 키 입력시 종료 
                    break 
                    
            self.cap.release()  
            cv2.destroyAllWindows()     
        except grpc._channel._Rendezvous as err :
            print(err)   
            

    def Capture(self): 
        """
        이 함수는 gRPC Streaming 를 위한 정보 입력과 발신 메세지를 처리합니다. 
        """
        time.sleep(1)
        while True:
            ret, frame = self.cap.read() # cap read 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            res = cv2.imencode('.jpg', frame)[1].tostring()
            msg = gateway_pb2.InvokeServiceRequest(Service= args.Handler, Input=res)
            self.dataQueue.put(msg)

        print("Streaing END!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This code is written for DCT Client about Bi-Streaming for Video')
    parser.add_argument('Handler', type=str,
            metavar='DCF Function name',
			help='Input to Use DCF Function')
    parser.add_argument('--video', type=str, default = int(0),
            metavar='Video file Name',
            help='Input to Use Video File Name \n if you use webcam, Just input 0')
    args = parser.parse_args()
    c = Client()
