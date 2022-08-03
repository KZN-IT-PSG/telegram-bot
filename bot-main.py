from aiogram import Bot
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from keras.models import load_model
import pickle

import os
import cv2

IMAGE_SIZE = 32
PATH = 'img/am.jpg'
bot = Bot(os.environ.get('TOKEN'))
dp = Dispatcher(bot)

names = ['Подшипник-3612 (Роликовый подшипник).', 'Игольчатый подшибник.', 'Шариковый подшипник.']

print("[INFO] loading network and label binarizer...")
model = load_model("model")
lb = pickle.loads(open("label.bin", "rb").read())


def _predict(path):
    image = cv2.imread(path)
    output = image.copy()
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    image = image.astype("float") / 255.0

    #image = image.flatten()
    image = image.reshape((-1, 32, 32, 3))

    preds = model.predict(image)

    i = preds.argmax(axis=1)[0]
    label = names[i]
    return label


@dp.message_handler(commands=['start', 'help'])
async def start_message(message):
    await bot.send_message(message.from_user.id, 'Отправь мне фото подшипника, и я определю его тип.')


@dp.message_handler(content_types=['photo'])
async def photo_predict(message):
    await bot.send_message(message.from_user.id, 'Фото обрабатывается...')
    await message.photo[-1].download(PATH)
    await message.reply(_predict(PATH))


@dp.message_handler(content_types=['text'])
async def text_send(message):
    await bot.send_message(message.from_user.id, 'Отправьте, пожалуйста, фотографию.')

def main():
    executor.start_polling(dp, skip_updates=True)


if __name__ == '__main__':
    main()
