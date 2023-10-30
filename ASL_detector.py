import tensorflow 
import mediapipe as mp
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
    
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic

def draw_keypoints(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    
    
def keypoints_array(holistic_results):
    if holistic_results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in holistic_results.pose_landmarks.landmark]).flatten() 
    else: 
        pose = np.zeros(132)               
    if holistic_results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in holistic_results.face_landmarks.landmark]).flatten() 
    else:
        face = np.zeros(1404)              
    if holistic_results.left_hand_landmarks:   
        lh = np.array([[res.x, res.y, res.z] for res in holistic_results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)              
    if holistic_results.right_hand_landmarks: 
        rh = np.array([[res.x, res.y, res.z] for res in holistic_results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
                

cap = cv.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))
        
DATA_PATH = os.path.join('MP_data')
actions = np.array(['hello', 'thanks', 'i love you'])

no_sequences = 10 #Videos 5 of Left and 5 of right
sequence_frames = 30 #Frames

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_frames):
                _, frame = cap.read() 
                if _ == True:
                    frame.flags.writeable = False
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    holistic_results = holistic.process(frame)
                    frame.flags.writeable = True
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    
                    draw_keypoints(frame, holistic_results)  
                    
                    if frame_num == 0:
                        cv.putText(frame, 'COLLECTION BEGINS', (120, 200), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 5, cv.LINE_AA)
                        cv.putText(frame, 'Training frames for {} Video Number {}'.format(action, sequence), (15, 15), 
                                   cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1 , cv.LINE_AA)
                        cv.waitKey(2000)
                    else:
                         cv.putText(frame, 'Training frames for {} Video Number {}'.format(action, sequence), (15, 15), 
                                    cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1 , cv.LINE_AA)
                    cv.imshow('MediaPipe Hands', frame)
                    
                    
                    result_test = keypoints_array(holistic_results)
                    numpy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(numpy_path, result_test)
                    
                    if cv.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    print("Camera issues detected")
                    break
                
label_map = {label:num for num, label in enumerate(actions)}
            

print("Exiting program now!")
cap.release()
cv.destroyAllWindows()
    

