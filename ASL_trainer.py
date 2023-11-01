from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import to_categorical
import tensorflow 
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import accuracy_score



actions = np.array(['hello', 'thanks', 'i love you'])

DATA_PATH = os.path.join('MP_data')
actions = np.array(['hello', 'thanks', 'i love you'])

no_sequences = 10 #Videos 5 of Left and 5 of right
sequence_frames = 30 #Frames

label_map = {label:num for num, label in enumerate(actions)}


sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_frames):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
        
seq_array = np.array(sequences)
labels_array = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(seq_array, labels_array)

print("Data pre-processing complete")

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape = (30, 1662), activation= 'relu'))
model.add(LSTM(128, return_sequences=True, activation= 'relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0],  activation='softmax'))

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ['categorical_crossentropy'])

model.fit(seq_array, labels_array, epochs=1000, callbacks=[tb_callback])


## predicter 
res = model.predict(x_test)

y_val = np.argmax(y_test, axis=1)
y_pred = np.argmax(res, axis=1)

score = accuracy_score(y_val, y_pred )

print(y_val)
print(y_pred)
print(score)

model.save('action.h5')
model.load_weights('action.h5')

