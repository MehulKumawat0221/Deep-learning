
import numpy as np
import pandas as pd

# CNN ---> Convolutional Neural Networks
# It is used for image related processing.
# Image --> pixels-> (0-255)px --->
# 0PX --> black
# 255 PX ---> white
# black & white Images ----> height, width
#colorfull Images ---> height, width, color_channel(RGB)
# We can convert our images into grayscale images . 127.5 px

#Im1 = (120,130,3)
#Im2 = (130, 150, 3)
#padding --> Add zero in image matrix.
#pre-padding --> It will add zeros in starting of an matrix.
#post-padding --> it wil add zeros in the last of an matrixs.

a = np.random.randint(1,12,9).reshape(3,3)
a

# pooling layers --> this is regularization technique in whoch it will reduce overfitting
#from data.
#Max Pooling --->

#stride = 1 , filter/kernal = (2,2)
# step-1 [10,9,6,10] ---> [10] [6]
# step-2 [9,6,10,5] ----> [10] [5]
# step-3 [6,10,1,11]----> [11] [1]
# step-4 [10,5,11,1] ---> [11] [1]
# MaxPooling layer --> [10,10,11,11]
# MinPooling layer ----> [6,5,1,1]
# GlobalAveragePooling Layer

# Workflow of CNN --->

# Data Ingestion ----> Data Cleaning and Preprocessing -->
# Model Sequential(
# CNN layer1,
# CNN_layer2,
# Flatten layer,
# Dense layer
# )

# Model.compile(loss, optimizers, metrics)
# Model.training
# Model.Evaluation

import tensorflow as tf

from tensorflow.keras import datasets , layers, models
import matplotlib.pyplot as plt

# Load and split datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize Pixel values to be betwwen oand 1 train_images, test_images = train_images/255.0,

class_names = ['airplane','automobile', 'bird', 'cat','deer', 'dog','frog','horse','ship', 'truck']

# Let's look at a one image. We change this to look at other images (img_index=1, 2,3....)
IMG_INDEX = 1
plt.imshow(train_images[IMG_INDEX], cmap = plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape =(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D (64, (3,3), activation = 'relu'))

model.summary()

model.add(layers.Flatten()) # We need to take these extracted features and add a way to classify them
#This is why we add the layers to our model .
model.add(layers.Dense (64, activation = 'relu'))
model.add(layers.Dense (10))

model.summary()

# We can see the flatten layer changes the shape of our data so that we can feed it to the 64 nodes dense layer
# followed by the final output layer followed by the final output layer of 10 neurons (one for each class).

#Now we will train and compile the model using the recommanded hyperparameters from tensorflow.
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history= model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)

"""#Now we will load the cats_vs_dogs dataset from the module tensorflow datasets.
#this datasets contains(images, labels)pairs where images have diffrent dimention

"""

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#split the data manually into 80% training and 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs',split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                                             with_info =True,
                                                            as_supervised= True)

get_label_name = metadata.features['label'].int2str

import matplotlib.pyplot as plt

#Create a function object that we can use to get labels
# Display 2 images from the datasets.
for image, label in raw_train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name (label))
# output ==> dog in different dimensions

IMG_SIZE = 160

import tensorflow as tf

def format_example(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
  return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

for img, label in raw_train.take(2):
  print("Original Shape:", img.shape)

for img, label in train.take(2):
  print("New Shape:", img.shape)

# Picking a Pre-trained Model ==>The model we are going to use as the convolutional base for our model is the
# MobileNet V2 developed at google .
# The model is trained on 1.4 million images and has 1000 different classes.
# We want to use this model but only its convolutional base.
# So, when we load in the model we will specify that we don't want to load the top (classfication)
# layer.
# We will tell the model what input shape to expect and to use the predetermined weights from
#imagenet(Google DataSet).

IMG_SIZE = 160

base_model = tf.keras.applications.MobileNetV2(
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    include_top = False,
    weights = 'imagenet'
  )

base_model.summary()

"""# Freezing the Base ==>
# The term freezing refers to disabling the training property of a layer
# It simply means we wont make any changes to the weights of any layers
# that are frozen during training .
# This is important as we don't want to change the convolutional lase that already has learned weights
"""

base_model.trainable = False

"""# The code base_model.trainable = False is used to set the trainable attribute of the base_model to False.
# In TensorFlow and Keras, the trainable attribute is a boolean flag that determines
# whether the weights of a model or a specific layer
# within the model should be updated during training or not.

# Adding our Classifier ==>
# Now that we have our base layer setup we can add the classfier
# Instead of flattening the feature map of the base layer we will use a global average pooling
# layer that will average the entire 5x5 area of each 2D features map and
# return to us a single 1280 element vector per filter .
"""

global_avarage_layer = tf.keras.layers.GlobalAveragePooling2D()

#finally we will add the prediction layer that will be a single dense neuron.
#we can do this because we only have two classes to predicted for.

prediction_layer = tf.keras.layers.Dense(1)

#now we will combine these layers together
model = tf.keras.Sequential([
    base_model,
    global_avarage_layer,
    prediction_layer
])

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate = base_learning_rate),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
              metrics = ['accuracy'])
# we

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

initial_epochs = 3
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)
acc =history.jistory['accuracy']
print(acc)
