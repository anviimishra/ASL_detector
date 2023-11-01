
import mediapipe as mp
import cv2 as cv
import h5py    
import numpy as np 
import keras  
from ASL_trainer import actions


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
model = keras.models.load_model('action.h5')
sequence = []
predictions = []
threshold = 0.7

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        _, frame = cap.read()
        if _ == True:
            frame.flags.writeable = False
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            holistic_results = holistic.process(frame)
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    
            draw_keypoints(frame, holistic_results) 
            
            keypoints =  keypoints_array(holistic_results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        print(actions[np.argmax(res)])
        else:
            print("Camera Isssues Faced")   

    
