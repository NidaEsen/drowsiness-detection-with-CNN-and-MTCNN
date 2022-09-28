import cv2
import dlib
import numpy as np
from imutils import face_utils
from facenet_pytorch import MTCNN
import threading
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from pygame import mixer


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

capture=cv2.VideoCapture(0)
if not (capture.isOpened()):

    print("Could not open video device")



model = load_model('__blinkModel2_.h5')
mixer.init()
sound = mixer.Sound('alarm.ogg')


        
class FaceDetector(object):
    
   
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn
        
    
    def _draw(self, frame, boxes, probs, landmarks):
        
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
               
    
               
        except:
            pass

        return frame
      
       
        
    def CNNpreprocessing(self,img):
      
        img=cv2.resize(img,(24,24));
        img = img.astype('float32')
        img=img/255
        img=np.expand_dims(img,axis=0)
        
        
        return img
    
    def find_eyes_and_crop(self,img,detected_faces):
       
        if len(detected_faces)>1:
            face=detected_faces[0]
        
        elif len(detected_faces)==0:
            return []
        else:
            [face]=detected_faces
            
       

        face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))  
 
        shape = predictor(img, face_rect)
        
    
        shape = face_utils.shape_to_np(shape)
       


        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        """
        img_=img
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img_, [leftEyeHull], -8, (0, 0, 255), 2)
        cv2.drawContours(img_, [rightEyeHull], -8, (0, 0, 255), 2)
        """
        #iki göz arası mesafe 
        distance_both_eye=abs(leftEye[3][0]-rightEye[0][0])//2
        #sol göz genişliği
        width_left_eye=abs(leftEye[3][0]-leftEye[0][0])

        
        height_left_eye= abs(leftEye[0][1]-leftEye[4][1]+distance_both_eye) if leftEye[4][1] < leftEye[3][1] else  abs(leftEye[0][1]-leftEye[3][1]+distance_both_eye)

        cropleft = img[leftEye[0][1]-height_left_eye:leftEye[0][1]+height_eye, leftEye[0][0]-width_left_eye:leftEye[0][0]+width_left_eye+distance_both_eye]
        
        #sağ göz genişliği
        distance_right_eye=abs(rightEye[3][0]-rightEye[0][0])
        #göz aralığı max değerini bulma
        height_right_eye= abs(rightEye[0][1]-rightEye[4][1]+distance_both_eye) if rightEye[4][1] < rightEye[3][1] else  abs(rightEye[0][1]-rightEye[3][1]+distance_both_eye)
       
        cropright = img[rightEye[0][1]-height_right_eye:rightEye[0][1]+height_right_eye, rightEye[0][0]-(distance_right_eye):rightEye[0][0]+(distance_right_eye+distance_both_eye)]
        
        return cropright,cropleft


    def run(self,capture):

        
        
        close_counter = blinks = mem_counter= 0
        state = ''
      
        while True:
            ret,frame= capture.read()
            if not ret:
                print("Can't receive frame...exiting")
                break
            
            try:    
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
                right_eye,left_eye= self.find_eyes_and_crop(frame,boxes)
                prediction= model.predict(self.CNNpreprocessing(right_eye))+model.predict(self.CNNpreprocessing(left_eye))/2.0
                print(prediction)
                       
                if prediction > 0.50 :
                    state = 'open'
                    close_counter = 0          
          
                else:
                    state = 'close'
                    close_counter += 1
                         
                if state == 'open' and mem_counter > 1:
                    sound.stop()
                
                    blinks += 1     
                 
                
                if prediction!= None:     
                    print(state)
            
                if state == 'close' and mem_counter>8 :
                
                    cv2.putText(frame, "UYUMA!", (10, 60),
 			          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
                    sound.play()
            
		 
                mem_counter = close_counter 

            except:
                pass
        
            cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "State: {}".format(state), (300, 30),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
            
            cv2.imshow('blinks counter', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                  break
        
        cv2.destroyAllWindows()
        del(capture)
        
mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)

t=threading.Thread(fcd.run(capture)) 
t.daemon=True
t.start()
                  
            
    


