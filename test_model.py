# импортируем необходимые пакеты
from keras.models import load_model
import argparse
import pickle
import cv2

# создаём парсер аргументов и передаём их
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#    help="path to input image we are going to classify")
#ap.add_argument("-m", "--model", required=True,
#    help="path to trained Keras model")
#ap.add_argument("-l", "--label-bin", required=True,
#    help="path to label binarizer")
#ap.add_argument("-w", "--width", type=int, default=28,
#    help="target spatial dimension width")
#ap.add_argument("-e", "--height", type=int, default=28,
#    help="target spatial dimension height")
#ap.add_argument("-f", "--flatten", type=int, default=-1,
#    help="whether or not we should flatten the image")
#args = vars(ap.parse_args())

width = height = 32
PATH = "3612.jpg"
 
# загружаем входное изображение и меняем его размер на необходимый
image = cv2.imread(PATH)
output = image.copy()
image = cv2.resize(image, (width, height))
 
# масштабируем значения пикселей к диапазону [0, 1]
image = image.astype("float") / 255.0

# проверяем, необходимо ли сгладить изображение и добавить размер
# пакета
if 1:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
 
# в противном случае мы работаем с CNN -- не сглаживаем изображение
# и просто добавляем размер пакета
else:
    image = image.reshape((1, image.shape[0], image.shape[1],
        image.shape[2]))


# загружаем модель и бинаризатор меток
print("[INFO] loading network and label binarizer...")
model = load_model("model")
lb = pickle.loads(open("label.bin", "rb").read())
 
# распознаём изображение
preds = model.predict(image)
 
# находим индекс метки класса с наибольшей вероятностью
# соответствия
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]


# рисуем метку класса + вероятность на выходном изображении
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    (0, 0, 255), 2)
 
# показываем выходное изображение
cv2.imshow("Image", output)
cv2.waitKey(0)
