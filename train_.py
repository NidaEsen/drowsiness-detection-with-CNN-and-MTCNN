from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation,Dropout
from tensorflow.keras import Sequential
import pandas as pd



def data_preprocessing(train_dir,test_dir):

    
    train_datagen_augmented=ImageDataGenerator(rescale=1/255.,
                                           rotation_range=0.2,
                                           #shear_range=0.2,
                                           #zoom_range=0.2,
                                           width_shift_range=0.2,
                                           height_shift_range=0.3,
                                           horizontal_flip=True)

    test_datagen=ImageDataGenerator(rescale=1/255.)


    train_data_augmented=train_datagen_augmented.flow_from_directory(train_dir,
                                                                 target_size=(24,24),
                                                                
                                                                 batch_size=32,
                                                                 
                                                                 class_mode="binary",
                                                                 shuffle=True)
    test_data=test_datagen.flow_from_directory(test_dir,
                                           target_size=(24,24),
                                           
                                           batch_size=32,
                                           class_mode="binary")
    
    return train_data_augmented,test_data



def plot_loss_curves(history1):
  loss=history1.history["loss"]
  val_loss=history1.history["val_loss"]
  accuracy=history1.history["acc"]
  val_accuracy=history1.history["val_acc"]
  print(pd.DataFrame(history1.history))
  epochs=range(len(history1 .history["loss"]))

  #plot loss
  plt.plot(epochs,loss,label="training_loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  #plot accuracy
  plt.figure()
  plt.plot(epochs,accuracy,label="training_accuracy")
  plt.plot(epochs,val_accuracy,label="val_acccuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend();
  
 
def build_model():  

    model=Sequential([
                  Conv2D(32,3,activation="relu",input_shape=(24,24,3)),
                  MaxPool2D(),
                  Conv2D(64,3,activation="relu"),
                  MaxPool2D(),
                  Conv2D(128,3,activation="relu"),
                  MaxPool2D(),
                  Dropout(0.25),
                  Flatten(),
                  Dense(256,activation="relu"),
                  Dense(512,activation="relu"),
                  Dense(1,activation="sigmoid")
                  
                  ])
    return model


def main():
   train_dir="C:/Users/Casper/Desktop/convolutional neural network/train"
   test_dir="C:/Users/Casper/Desktop/convolutional neural network/test"
   train_data_augmented,test_data=data_preprocessing(train_dir,test_dir)
   print(len(train_data_augmented),len(test_data))
   model=build_model()
   model.compile(loss="binary_crossentropy",
                 optimizer=Adam(),
                 metrics=["accuracy"])
   
   history=model.fit(train_data_augmented,
                    epochs=15,
                    steps_per_epoch=len(train_data_augmented),
                    validation_data=test_data,
                    validation_steps=len(test_data))  
  
   print(len(train_data_augmented),len(test_data))
   
   a=model.summary()
   
   
   plot_loss_curves(history)
   model.save('__blinkModel2_.h5')  
   
   
if __name__=='__main__':
    main()