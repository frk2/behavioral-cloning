
#Behavioral Cloning Project

In this project we were able to use the provided simulator and train our neural net to predit a steering angle based on the input data. Its kind of unbelievable to me that it actually worked out this way and how easy it was to make it learn something that would otherwise be unthinkable :) 

##Input preprocessing
This was by far the most important part of the project. The data was heavily biased towards 0 and the model would like to drive the car straight off cliffs. Small problem. Visually inspecting the data reveals the following:
![Original dataset, heavily zero biased](./original_dataset.png)

###Removing 0 bias
My data generator probabilistically rejects 0 valued steering values. It only accepts 5% of them, owing to the fact that they exceed normal data by about a factor of 20. This seems to work well.
```
if (steering_angle == 0. and np.random.uniform() > 0.05):
        continue
```

###Data Augmentation
To generate more data and generalize the model better - we use flipping and input from the left and right camera. The left and right camera help by generating a large 'view' of the track so the car will be able to recover if it goes slightly offcourse from the training set. Without the left and right cameras - recovery seems to not occur reliably. 
```
      img_choice = np.random.randint(3)
      if img_choice == 0:
          img_path = row.left
          steering_angle += OFF_CENTER_IMG
      elif img_choice == 1:
          img_path = row.center
      else:
          img_path = row.right
          steering_angle -= OFF_CENTER_IMG

      steering_angle = clamp(steering_angle, -1.0, 1.0)\
      
      image = cv2.imread(img_path.strip())
      image = preprocessImage(image, imgsize)
      if (np.random.randint(2) == 0):
        image = cv2.flip(image,1)
        steering_angle = -steering_angle
```
###Cropping
Cropping the top 60px and the bottom 40px helps generalize the model to track 2 a lot.This is mostly above the horizon so clouds and palm trees anyways :)

###The end result
![Balanced Data fed into model](./input_training_data.png)
We get a beautiful, normally distributed input dataset to our model. This would work well and looks WAY better than what we started with.

##The model

I tried many multiple models to figure out which one works. The models that are included in the model.py file and I tested were:
- My own model based off the keras labs we did. This was quite deep with > 10M parameters!
- Slightly modified nvidia model (https://arxiv.org/abs/1604.07316) 
- A fine tuned VGG16 model.

I found out that the nvidia model was most efficient and worked as well as the other 2 models with much less parameters (~1M). With VGG16 I had to use a very tiny learning rate for it to converge well. For the other 2 a default learning rate of 0.0001 seems to work well. I settled on the nvidia model for this.

To prevent overfitting, I've used a dropout layer after every FC layer. My input data set has about 14554*3*2 (3 cameras + flipping) = 87324 images. I feed this in the model over 50 epochs, each with 10,000 images.

##The approach
I used the udacity data as a 'trusted data source' to debug my code and get my model working. I started by using just a single camera with no augmentation other than flipping. I was able to make the car go around with just that data, given it was balanced properly. I then added on multiple cameras, cropping, more epochs and finally started training on my own track 1 driving data. Eventually I got it to generalize to track 2 by training on 2 laps (one forward, one reverse). 

###Make sure the generator isn't feeding garbage!
I started off the project with a minor bug in my code which cost me weeks! My generator was shoving out a lot of zero valued X,Y tuples which would obviously force my car to just drive straight all the time! Lesson learnt was to aggresively visualize the output of the generator and visualize the complete set of inputs through the generator to the model. This proved very useful. Adding a global variable to debug the generator is something i'll always do from now on!

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

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
