import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Activation, Cropping2D, Dropout, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# Set paths for the training data
base_dir = '/home/carnd/Data/'
img_dir = "IMG/"
csv_file_name = ['driving_log.csv', 'driving_log2.csv', 'driving_log3.csv', 'driving_log4.csv', 'driving_log5.csv']

# Load data from csv and images into training and label numpy arrays
df1 = pd.read_csv(base_dir+csv_file_name[0], header=None)
df2 = pd.read_csv(base_dir+csv_file_name[1], header=None)
df3 = pd.read_csv(base_dir+csv_file_name[2], header=None)
df4 = pd.read_csv(base_dir+csv_file_name[3], header=None)
df5 = pd.read_csv(base_dir+csv_file_name[4], header=None)
df = pd.concat([df1, df2, df3, df4, df5])
print("\nLoaded csv with %i rows" % len(df))

# Split into training and validation sets
data_train, data_valid = train_test_split(df, test_size=0.2)

# Generator for supplying data to the model in batches
def data_generator(data, batch_size, angle_bias=0.25):
    
    data = np.array(data)
    total_samples = len(data)
    
    while(True):
        
        # Shuffle the data before each epoch
        data = shuffle(data)
        
        # Obtain samples equal to batch size and process the images
        for offset in range(0, total_samples, batch_size):
            batch = data[offset:batch_size+offset]
            
            batch_images = []
            batch_labels = []
            
            # Load images and labels equal to the batch size only
            for row in batch:
                
                img_name = row[0].split("\\")[-1]
                img_C = cv2.cvtColor(cv2.imread(base_dir+img_dir+img_name), cv2.COLOR_BGR2RGB)
                label_C = float(row[3])
                
                img_name = row[1].split("\\")[-1]
                img_L = cv2.cvtColor(cv2.imread(base_dir+img_dir+img_name), cv2.COLOR_BGR2RGB)
                label_L = label_C + angle_bias
                
                img_name = row[2].split("\\")[-1]
                img_R = cv2.cvtColor(cv2.imread(base_dir+img_dir+img_name), cv2.COLOR_BGR2RGB)
                label_R = label_C - angle_bias
                
                # Append to the batch data
                batch_images.append(img_C)
                batch_images.append(img_L)
                batch_images.append(img_R)
                
                batch_labels.append(label_C)
                batch_labels.append(label_L)
                batch_labels.append(label_R)
                
                # Flip the image
                flip_img_C = cv2.flip(img_C, 1)
                flip_label_C = label_C * (-1)
                
                flip_img_L = cv2.flip(img_L, 1)
                flip_label_L = label_L * (-1)
                
                flip_img_R = cv2.flip(img_R, 1)
                flip_label_R = label_R * (-1)
                
                # Append the augmented data to the batch data
                batch_images.append(flip_img_C)
                batch_images.append(flip_img_L)
                batch_images.append(flip_img_R)
                
                batch_labels.append(flip_label_C)
                batch_labels.append(flip_label_L)
                batch_labels.append(flip_label_R)
                
                          
            X = np.array(batch_images)
            y = np.array(batch_labels)

            yield shuffle(X, y)
			
# Create generators for training and validation
train_generator = data_generator(data_train, batch_size=32)
valid_generator = data_generator(data_valid, batch_size=32)

# Model Definition
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

model.add(Convolution2D(8, 5, 5, border_mode='valid', subsample=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation(activation='relu'))

model.add(Convolution2D(6, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation(activation='relu'))

model.add(Convolution2D(6, 3, 3, border_mode='same', subsample=(1,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation(activation='relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

adam_opt = Adam(lr=0.005)
model.compile(loss='mse', optimizer=adam_opt)
filepath="model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', patience=3)
print("Model compiled\n")

# Train the model
history = model.fit_generator(train_generator, callbacks=[checkpoint, earlystop], 
			samples_per_epoch=len(data_train), validation_data=valid_generator, 
			nb_val_samples=len(data_valid), nb_epoch=40)
			