from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers import Conv2D,MaxPooling2D



trainData='C:\\Users\\pcpc\\Desktop\\Duboko ucenje\\DLprojekat\\train'
valData='C:\\Users\\pcpc\\Desktop\\Duboko ucenje\\DLprojekat\\test'

trainDataProc = ImageDataGenerator(
					rescale=1./255,
					rotation_range=25,
					shear_range=0.4,
					zoom_range=0.2,
					horizontal_flip=True)
valDataProc = ImageDataGenerator(rescale=1./255)
trainGenerator = trainDataProc.flow_from_directory(
					trainData,
					color_mode='grayscale',
					target_size=(48,48),
					batch_size=64,
					class_mode='categorical',
					shuffle=True)
validationGenerator = valDataProc.flow_from_directory(
							valData,
							color_mode='grayscale',
							target_size=(48,48),
							batch_size=64,
							class_mode='categorical',
							shuffle=True)

model = Sequential()
model.add(Conv2D(16,kernel_size=(2, 2),activation='relu',input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,kernel_size=(2,2),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(trainGenerator,
                steps_per_epoch=28709//64,
                epochs=100,
                validation_data=validationGenerator,
                validation_steps=7178//64)
model.save('emocionalniModel.h5')