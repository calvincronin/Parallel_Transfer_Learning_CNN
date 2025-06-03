import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
import time
import imagesize
import kaggle

# import tensorflow/keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Rescaling
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report
import random

print('Python version')
print (sys.version)

print('Tensorflow version')
print(tf.__version__)

# set output line width
np.set_printoptions(linewidth=200, precision=3)
# set default for keras to float
tf.keras.backend.set_floatx('float64')

tf.autograph.set_verbosity(0)

StartTime = time.time()

kaggle.api.authenticate()
kaggle.api.dataset_download_files('Paralell_Transfer_Learning_CNN', path='.', unzip=True)

def cord(hotvec):
    x0 = hotvec[0]
    x1 = hotvec[1]
    if x0 > x1:
        return('cat')
    elif x0 < x1:
        return('dog')
    else:
        print('samesameissue')
        return(random.choice(['cat','dog']))


catsM1 = tf.keras.utils.image_dataset_from_directory(
    'cats0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats', 'dogs'],
    batch_size=None,
    seed=6
)

os.mkdir('cats0/foxes')

catsM2 = tf.keras.utils.image_dataset_from_directory(
    'cats0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats', 'dogs', 'foxes'],
    batch_size=None,
    seed=6
)

os.rmdir('cats0/foxes')
os.mkdir('cats0/jags')
os.mkdir('cats0/wolves')

catsM3 = tf.keras.utils.image_dataset_from_directory(
    'cats0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats', 'dogs', 'jags', 'wolves'],
    batch_size=None,
    seed=6
)

dogsM1 = tf.keras.utils.image_dataset_from_directory(
    'dogs0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats','dogs'],
    batch_size=None,
    seed=6
)

os.mkdir('dogs0/foxes')

dogsM2 = tf.keras.utils.image_dataset_from_directory(
    'dogs0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats','dogs','foxes'],
    batch_size=None,
    seed=6
)

os.rmdir('dogs0/foxes')
os.mkdir('dogs0/jags')
os.mkdir('dogs0/wolves')

dogsM3 = tf.keras.utils.image_dataset_from_directory(
    'dogs0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats','dogs', 'jags', 'wolves'],
    batch_size=None,
    seed=6
)

foxesM2 = tf.keras.utils.image_dataset_from_directory(
    'foxes0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats', 'dogs', 'foxes'],
    batch_size=None,
    seed=6
)

jagsM3 = tf.keras.utils.image_dataset_from_directory(
    'jags0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats','dogs', 'jags', 'wolves'],
    batch_size=None,
    seed=6
)

wolvesM3 = tf.keras.utils.image_dataset_from_directory(
    'wolves0',
    labels='inferred',
    label_mode='categorical',
    class_names=['cats','dogs', 'jags', 'wolves'],
    batch_size=None,
    seed=6
)



cattrainM1 = catsM1.take(1000)
cattestvalM1 = catsM1.skip(1000)
cattestM1 = cattestvalM1.take(250)
catvalM1 = cattestvalM1.skip(250).take(250)

cattrainM2 = catsM2.take(1000)
cattestvalM2 = catsM2.skip(1000)
cattestM2 = cattestvalM2.take(250)
catvalM2 = cattestvalM2.skip(250).take(250)

cattrainM3 = catsM3.take(1000)
cattestvalM3 = catsM3.skip(1000)
cattestM3 = cattestvalM3.take(250)
catvalM3 = cattestvalM3.skip(250).take(250)

dogtrainM1 = dogsM1.take(1000)
dogtestvalM1 = dogsM1.skip(1000)
dogtestM1 = dogtestvalM1.take(250)
dogvalM1 = dogtestvalM1.skip(250).take(250)

dogtrainM2 = dogsM2.take(1000)
dogtestvalM2 = dogsM2.skip(1000)
dogtestM2 = dogtestvalM2.take(250)
dogvalM2 = dogtestvalM2.skip(250).take(250)

dogtrainM3 = dogsM3.take(1000)
dogtestvalM3 = dogsM3.skip(1000)
dogtestM3 = dogtestvalM3.take(250)
dogvalM3 = dogtestvalM3.skip(250).take(250)

foxtrain = foxesM2.take(500)
foxval = foxesM2.skip(500)

jagtrain = jagsM3.take(500)
jagval = jagsM3.skip(500)

wolftrain = wolvesM3.take(500)
wolfval = wolvesM3.skip(500)


m1train = cattrainM1.concatenate(dogtrainM1)
m1train = m1train.shuffle(2000, seed=6)
m1train = m1train.batch(32)

m1val = catvalM1.concatenate(dogvalM1)
m1val = m1val.shuffle(600, seed=6)
m1val = m1val.batch(32)

m1test = cattestM1.concatenate(dogtestM1)
m1test = m1test.shuffle(600, seed=6)
m1test = m1test.batch(32)


m2train = cattrainM2.concatenate(dogtrainM2).concatenate(foxtrain)
m2train = m2train.shuffle(2500, seed=6)
m2train = m2train.batch(32)

m2val = catvalM2.concatenate(dogvalM2).concatenate(foxval)
m2val = m2val.shuffle(750, seed=6)
m2val = m2val.batch(32)

m2test = cattestM2.concatenate(dogtestM2)
m2test = m2test.shuffle(600, seed=6)
m2test = m2test.batch(32)


m3train = cattrainM3.concatenate(dogtrainM3).concatenate(jagtrain).concatinate(wolftrain)
m3train = m3train.shuffle(4000, seed=6)
m3train = m3train.batch(32)

m3val = catvalM3.concatenate(dogvalM3).concatenate(jagval).concatenate(wolfval)
m3val = m3val.shuffle(900, seed=6)
m3val = m3val.batch(32)

m3test = cattestM3.concatenate(dogtestM3)
m3test = m3test.shuffle(600, seed=6)
m3test = m3test.batch(32)




model1 = Sequential()

# Feature Learning Layers
model1.add(Conv2D(32,                  # Number of filters/Kernels
                    (3,3),               # Size of kernels (3x3 matrix)
                    strides = 1,         # Step size for sliding the kernel across the input (1 pixel at a time).
                    padding = 'same',    # 'Same' ensures that the output feature map has the same dimensions as the input by padding zeros around the input.
                input_shape = (256,256,3) # Input image shape
                ))

model1.add(Activation('relu'))# Activation function
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model1.add(Dropout(0.2))

model1.add(Conv2D(64, (5,5), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model1.add(Dropout(0.2))

model1.add(Conv2D(128, (3,3), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model1.add(Dropout(0.3))

model1.add(Conv2D(256, (5,5), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model1.add(Dropout(0.3))

model1.add(Conv2D(512, (3,3), padding = 'same'))
model1.add(Activation('relu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model1.add(Dropout(0.3))

# Flattening tensors
model1.add(Flatten())

# Fully-Connected Layers
model1.add(Dense(2048))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))

# Output Layer
model1.add(Dense(2, activation = 'softmax')) # Classification layer


model2 = Sequential()

# Feature Learning Layers
model2.add(Conv2D(32,                  # Number of filters/Kernels
                    (3,3),               # Size of kernels (3x3 matrix)
                    strides = 1,         # Step size for sliding the kernel across the input (1 pixel at a time).
                    padding = 'same',    # 'Same' ensures that the output feature map has the same dimensions as the input by padding zeros around the input.
                input_shape = (256,256,3) # Input image shape
                ))

model2.add(Activation('relu'))# Activation function
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model2.add(Dropout(0.2))

model2.add(Conv2D(64, (5,5), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model2.add(Dropout(0.2))

model2.add(Conv2D(128, (3,3), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model2.add(Dropout(0.3))

model2.add(Conv2D(256, (5,5), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model2.add(Dropout(0.3))

model2.add(Conv2D(512, (3,3), padding = 'same'))
model2.add(Activation('relu'))
model2.add(BatchNormalization())
model2.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model2.add(Dropout(0.3))

# Flattening tensors
model2.add(Flatten())

# Fully-Connected Layers
model2.add(Dense(2048))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))

# Output Layer
model2.add(Dense(3, activation = 'softmax')) # Classification layer


model3 = Sequential()

# Feature Learning Layers
model3.add(Conv2D(32,                  # Number of filters/Kernels
                    (3,3),               # Size of kernels (3x3 matrix)
                    strides = 1,         # Step size for sliding the kernel across the input (1 pixel at a time).
                    padding = 'same',    # 'Same' ensures that the output feature map has the same dimensions as the input by padding zeros around the input.
                input_shape = (256,256,3) # Input image shape
                ))

model3.add(Activation('relu'))# Activation function
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model3.add(Dropout(0.2))

model3.add(Conv2D(64, (5,5), padding = 'same'))
model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model3.add(Dropout(0.2))

model3.add(Conv2D(128, (3,3), padding = 'same'))
model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model3.add(Dropout(0.3))

model3.add(Conv2D(256, (5,5), padding = 'same'))
model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model3.add(Dropout(0.3))

model3.add(Conv2D(512, (3,3), padding = 'same'))
model3.add(Activation('relu'))
model3.add(BatchNormalization())
model3.add(MaxPooling2D(pool_size = (2,2), padding = 'same'))
model3.add(Dropout(0.3))

# Flattening tensors
model3.add(Flatten())

# Fully-Connected Layers
model3.add(Dense(2048))
model3.add(Activation('relu'))
model3.add(Dropout(0.5))

# Output Layer
model3.add(Dense(4, activation = 'softmax')) # Classification layer


model1.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001), # 1e-4
              loss = 'categorical_crossentropy', # Ideal for multiclass tasks
              metrics = ['accuracy'])

model2.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001), # 1e-4
              loss = 'categorical_crossentropy', # Ideal for multiclass tasks
              metrics = ['accuracy'])

model3.compile(optimizer = tf.keras.optimizers.RMSprop(0.0001), # 1e-4
              loss = 'categorical_crossentropy', # Ideal for multiclass tasks
              metrics = ['accuracy'])


checkpoint1 = ModelCheckpoint('best_model1.h5',
                            monitor = 'val_accuracy',
                            save_best_only = True)

checkpoint2 = ModelCheckpoint('best_model2.h5',
                            monitor = 'val_accuracy',
                            save_best_only = True)

checkpoint3 = ModelCheckpoint('best_model3.h5',
                            monitor = 'val_accuracy',
                            save_best_only = True)



# Training and Testing Models
try:
    history1 = model1.fit(
        m1train, epochs = 20,
        validation_data = m1val,
        callbacks = [checkpoint1])
except Exception as e:
    print("An error occurred:", e)

modelName = 'Baseline Model'
numEpochs = 20
batchSize = 32
validationSplit = 0.23
elapsed = 5
# note: if loss curve starts to reverse direction we could be overfitting model
fig = plt.figure(figsize=(round(numEpochs/1),8))
ax1 = fig.add_subplot(111)
# get max values for annotation
ind_xta = np.argmax(history1.history['accuracy'])
ind_xva = np.argmax(history1.history['val_accuracy'])
ind_xtl = np.argmin(history1.history['loss'])
ind_xvl = np.argmin(history1.history['val_loss'])
plt.title('\n\nModel #'+str(modelName)+'-'+' Training and Validation Metrics'+
    '\n\nTraining Batch Size='+str(batchSize)+' '+
    'Epochs='+str(numEpochs)+ ' '+
    'Validation Split '+str(validationSplit)+' '+
    'Max Training Accuracy='+str("%.3f" % history1.history['accuracy'][ind_xta])+' ' +
    'Min Training Loss='+str("%.3f" % history1.history['loss'][ind_xtl])+'\n'+
    'Max Validation Accuracy='+str("%.3f" % history1.history['val_accuracy'][ind_xva])+' ' +
    'Min Validation Loss='+str("%.3f" % history1.history['val_loss'][ind_xvl])+'\n',size=14)
ax1.plot(history1.history['accuracy'], color='green', linestyle='-',linewidth=2)
plt.plot(history1.history['val_accuracy'], color='lime', linestyle='--',linewidth=2)

ax1.set_xlabel('\nEpoch', size=15)
ax1.set_ylabel('Accuracy\n\nSolid=Training Dashed=Validation\n', color='green',size=15)
ax1.set_xlim(right=numEpochs)
plt.xticks(np.arange(len(history1.history['accuracy'])), np.arange(1, len(history1.history['accuracy'])+1))

plt.plot(ind_xta, history1.history['accuracy'][ind_xta], 'gx', color='green', markersize=8, markeredgewidth=2)
plt.plot(ind_xva, history1.history['val_accuracy'][ind_xva], 'gx', color='lime', markersize=8, markeredgewidth=2)

for tl in ax1.get_yticklabels():
    tl.set_color('green')
    tl.set_fontsize(12)

ax2 = ax1.twinx()
ax2.plot(history1.history['loss'], color='darkred', linestyle='-',linewidth=2)
plt.plot(history1.history['val_loss'], color='red', linestyle='--',linewidth=2)

ax2.set_ylabel('Loss\n\nSolid=Training Dashed=Validation\n', color='darkred',size=15, rotation=270, labelpad=70)
plt.plot(ind_xtl, history1.history['loss'][ind_xtl], 'rx', color='darkred', markersize=8, markeredgewidth=2)
plt.plot(ind_xvl, history1.history['val_loss'][ind_xvl], 'rx', color='red', markersize=8, markeredgewidth=2)

for tl in ax2.get_yticklabels():
    tl.set_color('darkred')
    tl.set_fontsize(12)

fig.show()


preds1 = model1.predict(m1test)
y1 = np.concatenate([y for x, y in m1test], axis=0)

ytrue1 = np.array([cord(hv) for hv in y1])
ypred1 = np.array([cord(hv) for hv in preds1])

cr1 = classification_report(ytrue1, ypred1, output_dict=True)
df1 = pd.DataFrame(cr1).transpose()
df1



modelName = 'Fox-Plus Model'
try:
    history2 = model2.fit(
        m2train, epochs = 20,
        validation_data = m2val,
        callbacks = [checkpoint2])
except Exception as e:
    print("An error occurred:", e)


fig = plt.figure(figsize=(round(numEpochs/1),8))
ax1 = fig.add_subplot(111)
# get max values for annotation
ind_xta = np.argmax(history2.history['accuracy'])
ind_xva = np.argmax(history2.history['val_accuracy'])
ind_xtl = np.argmin(history2.history['loss'])
ind_xvl = np.argmin(history2.history['val_loss'])
plt.title('\n\nModel #'+str(modelName)+'-'+' Training and Validation Metrics'+
    '\n\nTraining Batch Size='+str(batchSize)+' '+
    'Epochs='+str(numEpochs)+ ' '+
    'Validation Split '+str(validationSplit)+' '+
    'Duration '+str(elapsed)+' secs\n'+
    'Max Training Accuracy='+str("%.3f" % history2.history['accuracy'][ind_xta])+' ' +
    'Min Training Loss='+str("%.3f" % history2.history['loss'][ind_xtl])+'\n'+
    'Max Validation Accuracy='+str("%.3f" % history2.history['val_accuracy'][ind_xva])+' ' +
    'Min Validation Loss='+str("%.3f" % history2.history['val_loss'][ind_xvl])+'\n',size=14)
ax1.plot(history2.history['accuracy'], color='green', linestyle='-',linewidth=2)
plt.plot(history2.history['val_accuracy'], color='lime', linestyle='--',linewidth=2)

ax1.set_xlabel('\nEpoch', size=15)
ax1.set_ylabel('Accuracy\n\nSolid=Training Dashed=Validation\n', color='green',size=15)
ax1.set_xlim(right=numEpochs)
plt.xticks(np.arange(len(history2.history['accuracy'])), np.arange(1, len(history2.history['accuracy'])+1))

plt.plot(ind_xta, history2.history['accuracy'][ind_xta], 'gx', color='green', markersize=8, markeredgewidth=2)
plt.plot(ind_xva, history2.history['val_accuracy'][ind_xva], 'gx', color='lime', markersize=8, markeredgewidth=2)

for tl in ax1.get_yticklabels():
    tl.set_color('green')
    tl.set_fontsize(12)

ax2 = ax1.twinx()
ax2.plot(history2.history['loss'], color='darkred', linestyle='-',linewidth=2)
plt.plot(history2.history['val_loss'], color='red', linestyle='--',linewidth=2)

ax2.set_ylabel('Loss\n\nSolid=Training Dashed=Validation\n', color='darkred',size=15, rotation=270, labelpad=70)
plt.plot(ind_xtl, history2.history['loss'][ind_xtl], 'rx', color='darkred', markersize=8, markeredgewidth=2)
plt.plot(ind_xvl, history2.history['val_loss'][ind_xvl], 'rx', color='red', markersize=8, markeredgewidth=2)

for tl in ax2.get_yticklabels():
    tl.set_color('darkred')
    tl.set_fontsize(12)

fig.show()

preds2 = model2.predict(m2test)
y2 = np.concatenate([y for x, y in m2test], axis=0)

ytrue2 = np.array([cord(hv) for hv in y2])
ypred2 = np.array([cord(hv) for hv in preds2])

cr2 = classification_report(ytrue2, ypred2, output_dict=True)

df2 = pd.DataFrame(cr2).transpose()
df2



modelName = 'Jag-Wolf-Plus Model'
try:
    history3 = model3.fit(
        m3train, epochs = 20,
        validation_data = m3val,
        callbacks = [checkpoint3])
except Exception as e:
    print("An error occurred:", e)

fig = plt.figure(figsize=(round(numEpochs/1),8))
ax1 = fig.add_subplot(111)
# get max values for annotation
ind_xta = np.argmax(history3.history['accuracy'])
ind_xva = np.argmax(history3.history['val_accuracy'])
ind_xtl = np.argmin(history3.history['loss'])
ind_xvl = np.argmin(history3.history['val_loss'])
plt.title('\n\nModel #'+str(modelName)+'-'+' Training and Validation Metrics'+
    '\n\nTraining Batch Size='+str(batchSize)+' '+
    'Epochs='+str(numEpochs)+ ' '+
    'Validation Split '+str(validationSplit)+' '+
    'Duration '+str(elapsed)+' secs\n'+
    'Max Training Accuracy='+str("%.3f" % history3.history['accuracy'][ind_xta])+' ' +
    'Min Training Loss='+str("%.3f" % history3.history['loss'][ind_xtl])+'\n'+
    'Max Validation Accuracy='+str("%.3f" % history3.history['val_accuracy'][ind_xva])+' ' +
    'Min Validation Loss='+str("%.3f" % history3.history['val_loss'][ind_xvl])+'\n',size=14)
ax1.plot(history3.history['accuracy'], color='green', linestyle='-',linewidth=2)
plt.plot(history3.history['val_accuracy'], color='lime', linestyle='--',linewidth=2)

ax1.set_xlabel('\nEpoch', size=15)
ax1.set_ylabel('Accuracy\n\nSolid=Training Dashed=Validation\n', color='green',size=15)
ax1.set_xlim(right=numEpochs)
plt.xticks(np.arange(len(history3.history['accuracy'])), np.arange(1, len(history3.history['accuracy'])+1))

plt.plot(ind_xta, history3.history['accuracy'][ind_xta], 'gx', color='green', markersize=8, markeredgewidth=2)
plt.plot(ind_xva, history3.history['val_accuracy'][ind_xva], 'gx', color='lime', markersize=8, markeredgewidth=2)

for tl in ax1.get_yticklabels():
    tl.set_color('green')
    tl.set_fontsize(12)

ax2 = ax1.twinx()
ax2.plot(history3.history['loss'], color='darkred', linestyle='-',linewidth=2)
plt.plot(history3.history['val_loss'], color='red', linestyle='--',linewidth=2)

ax2.set_ylabel('Loss\n\nSolid=Training Dashed=Validation\n', color='darkred',size=15, rotation=270, labelpad=70)
plt.plot(ind_xtl, history3.history['loss'][ind_xtl], 'rx', color='darkred', markersize=8, markeredgewidth=2)
plt.plot(ind_xvl, history3.history['val_loss'][ind_xvl], 'rx', color='red', markersize=8, markeredgewidth=2)

for tl in ax2.get_yticklabels():
    tl.set_color('darkred')
    tl.set_fontsize(12)

fig.show()

preds3 = model3.predict(m3test)
y3 = np.concatenate([y for x, y in m3test], axis=0)

ytrue3 = np.array([cord(hv) for hv in y3])
ypred3 = np.array([cord(hv) for hv in preds3])

cr3 = classification_report(ytrue3, ypred3, output_dict=True)

df3 = pd.DataFrame(cr3).transpose()
df3
