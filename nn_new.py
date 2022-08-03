import os

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import pickle
import cv2
import random

print("[INFO] loading images...")
data = []
labels = []
PATH = "D:/IT-PSG_Projects/telegram-bot/podshipniks"
 
class SmallerVGGNet:
	@staticmethod
	def build(width, height, depth, classes, finalAct="softmax"):
		# инициализируем модель вместе с формой входных
		# данных "channels last" и размерностью каналов
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
 
		# если мы используем "channels first", обновляем форму
		# входных данных и размерность каналов
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
        
        # CONV =&gt; RELU =&gt; POOL
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(3, 3)))
		model.add(Dropout(0.25))
        # (CONV => RELU) * 2 => POOL
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
 
		# (CONV => RELU) * 2 => POOL
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
        # первый (и единственный) набор слоёв FC =&gt; RELU
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(Dropout(0.5))
 
		# используйте *softmax* активацию для классификации 
		# по одной метке и *sigmoid* по нескольким
		model.add(Dense(classes))
		model.add(Activation(finalAct))
 
		# возвращаем сетевую архитектуру
		return model

# берём пути к изображениям и рандомно перемешиваем
imagePaths = sorted(list(paths.list_images("podshipniks")))
random.seed(42)
random.shuffle(imagePaths)
 
# цикл по изображениям
for imagePath in imagePaths:
    # загружаем изображение, меняем размер на 32x32 пикселей (без учёта
    # соотношения сторон), сглаживаем его в 32x32x3=3072 пикселей и
    # добавляем в список
    try:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32,32))
        data.append(image)
    except Exception as e:
        print(imagePath)
        exit()

    # извлекаем метку класса из пути к изображению и обновляем
    # список меток
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label) 
    
# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 75% 
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# конвертируем метки из целых чисел в векторы (для 2х классов при 
# бинарной классификации вам следует использовать функцию Keras 
# “to_categorical” вместо “LabelBinarizer” из scikit-learn, которая
# не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

EPOCHS = 150
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (32, 32, 3)

print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_),
	finalAct="sigmoid")
 
# инициализируем оптимизатор
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# обучаем нейросеть
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
 
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("loss")

# сохраняем модель и бинаризатор меток на диск
print("[INFO] serializing network and label binarizer...")
model.save("model")
f = open("label.bin", "wb")
f.write(pickle.dumps(lb))
f.close()