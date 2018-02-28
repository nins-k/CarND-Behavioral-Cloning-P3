# Udacity Self-Driving Car Nanodegree

## Term 1 : Project 03 - **Behavioral Cloning**

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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
The driving behaviour of the car is seen to be greatly dependent on the hardware on which the simulator is being run. On testing this on a workstation without a GPU, the **drive.py** file had to be edited to change the **set_speed** variable to **4** instead of the default **9** (on line 47).

On running the simulation on a laptop with a GPU, it works fine with the default value of 9. In the submission, the **drive.py** file is included as-is as provided by Udacity without any modification.

#### 3. Submission code is usable and readable

The **model.py** contains all the code required to load the training data, create a model and train it using Keras and saving the model to disk. The code is commented with explanation wherever required.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model consists of **three convolutional layers** and **three fully connected layers**. The model was chosen to be sufficiently complex to navigate the tracks, especially given that collecting accurate data was very challenging - (I decided to use only my own data and did not train on the Udacity provided data).

Below is a Keras generated summary representation of the model:

![model_summary](markdown_images/01_model_summary.JPG)



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
