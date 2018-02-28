# Udacity Self-Driving Car Nanodegree

## Term 1 : Project 03 - **Behavioral Cloning**

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) - script to create and train the model.
* [drive.py](./drive.py) - for driving the car in autonomous mode.
* [model.h5](./model.h5) - containing a trained convolution neural network 
* [writeup_report.md](./writeup_report.md) - THIS file is the Write Up report/summary.
* [P3_Model_Generator.ipynb](./P3_Model_Generator.ipynb) - Additionally, I have included the IPython Notebook I used to write the code.

#### 2. Submission includes functional code
The below code can be used to run the simulator in Autonomous mode with my trained model.
```sh
python drive.py model.h5
```
The driving behaviour of the car is seen to be greatly dependent on the hardware on which the simulator is being run. While testing this on a workstation without a GPU, the **drive.py** file had to be edited to change the **set_speed** variable to **4** instead of the default **9** (on line 47).

While running the simulator on a laptop with a GPU, it works fine with the default value of 9. In the submission, the **drive.py** file is included as-is as provided by Udacity without any modification.

#### 3. Submission code is usable and readable

The **model.py** contains all the code required to load the training data, create a model and train it using Keras and saving the model to disk. The code is commented with explanation wherever required.

<hr>

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of **three convolutional layers** and **three fully connected layers** apart from the single-neuron output layer. The model was chosen to be sufficiently complex to navigate the tracks, especially given that collecting accurate data was very challenging - (I decided to use only my own data and did not train on the Udacity provided data). A **generator** was used to load and augment the data to reduce the memory requirements.

A detailed explanation of the model with visualization is provided in the following sections.

#### 2. Attempts to reduce overfitting in the model

1. **Dropout** is implemented in the three fully connected layers.
2. **Augmentation** is used by including the left and right camera images for training with a small adjustment of the steering angle.
3. Further, **Flipping** is done on all the images to increase the data and better generalize it.

#### 3. Model parameter tuning

The model uses an Adam optimizer but the initial learning rate was explicitly defined to be **0.005** which was experimentally found to help the model converge better.
```python
adam_opt = Adam(lr=0.005)
```

#### 4. Appropriate training data

The Udacity training data was not used. I wanted to use my own data which I have recorded by attempting to drive along the center. Left and Right camera images were also used for recovery and flipping was done on all images to to counter turn-bias.

<hr>

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a sufficiently strong model which was reporting low loss on both; training and validation data set. 

On the Epoch with the best performance, the loss was reported as below:

|Loss|Value|
|----|----|
|Training Loss|0.0310|
|Validation Loss|0.0327|

The challenge was training on accurate data. For a few curves, I had to record additional data to improve the model.

Below are a few key features of the model and training:

* Lambda Layer (Normalization)
```python
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
```
* Cropping Layer
```python
model.add(Cropping2D(cropping=((50,20), (0,0))))
``` 
* Early Stopping Callback<sup>[[2]](https://keras.io/callbacks/)</sup>
```python
earlystop = EarlyStopping(monitor='val_loss', patience=3)
```
* Checkpoint Callback<sup>[[2]](https://keras.io/callbacks/)</sup>
```python
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)
```

#### 2. Final Model Architecture

Below is a diagrammatic representation of the model architecture generated using draw_convnet.<sup>[[1]](https://github.com/gwding/draw_convnet)

![model_summary](markdown_images/02_model_arch_diagram.png)

(It is better to download and view the image locally as it is difficult to read the details within the browser)

Note that this diagram does not illustrate the below listed features:
* Lambda layer for normalization
* Cropping layer
* Activation functions
* Dropout
* Loss Function

All details are included in the Keras generated summary representation of the model:

![model_summary](markdown_images/01_model_summary.JPG)

Below is the code section which compiles and trains the model:
```python
adam_opt = Adam(lr=0.005)
model.compile(loss='mse', optimizer=adam_opt)
history = model.fit_generator(train_generator, callbacks=[checkpoint, earlystop], 
			samples_per_epoch=len(data_train), validation_data=valid_generator, 
			nb_val_samples=len(data_valid), nb_epoch=40)
```

#### 3. Creation of the Training Set & Training Process

The training data was generated by driving on *Track 1* only; adhering to the center of the road as much as possible. Due to hardware limitations, this was especially challenging.

Below is an example of a captured training image from the center camera and its left and right camera counterparts.
![Sample Center Image](markdown_images/04_sample_center.jpg)

![Sample Left Image](markdown_images/03_sample_left.jpg)![Sample Right Image](markdown_images/05_sample_right.jpg)

For augmentation, the left and right images were also included in the training set by adding an angle bias: **+25°** for the left camera images and **-25°** for the right camera images to act as recovery data.

```python
# Add bias to left image label
img_name = row[1].split("\\")[-1]
img_L = cv2.cvtColor(cv2.imread(base_dir+img_dir+img_name), cv2.COLOR_BGR2RGB)
                label_L = label_C + angle_bias
```

Further, all three images were flipped and their steering angle sign inverted to double the amount of training data. This also helped to counter any bias for anti-clockwise driving. All of this was done within the body of the generator.

```python
 # Flip the image
 flip_img_C = cv2.flip(img_C, 1)
 flip_label_C = label_C * (-1)
```

Below is the above center image after flipping

![Flipped Center Image](markdown_images/06_sample_center.jpg)

As mentioned above, the model had some trouble maneuvering the curves at the end of the lap for which additional data had to be recorded.


The **EarlyStopping** Keras callback helped to use the most optimum model.

A warning is shown during training because the number of samples is not exactly divisible by the batch size. But this has no effect on the performance and can be ignored.<sup>[[3]](https://stackoverflow.com/questions/41789961/keras-warning-epoch-comprised-more-than-samples-per-epoch-samples)</sup>

<hr>

#### 4. Appendix

##### Quick Code Reference ##### 

|Feature|Description|Line Numbers (model.py)|
|-------|-----------|------------|
|Read CSV|Read driving log CSV files using Pandas|18 - 24|
|Split Data|Use sklearn to split data|28|
|Generator|Read and augment images|31 - 95|
|Shuffle-1|Shuffle at the start of the epoch|39|
|Image Read|Read and convert to RGB|52|
|Flip|Flip the image using OpenCV|73|
|Shuffle-2|Shuffle the samples WITHIN the batch|95|
|Optimizer|Adam with explicit learning rate|127|
|Checkpoint|Model checkpoint for save_best_only|130|
|EarlyStopping|Early stop callback for Keras|131|
|Training|Train with fit_generator|135|

