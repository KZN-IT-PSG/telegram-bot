Telegram-bot for determine the type of bearing.

Necessary packages for the bot to work:
aiogram;
tensorflow;
opencv-python.

Setting up the environment for the bot:
Use folowing commands:

python -m venv (your venv name)

(your venv name)\Scripts\activate

pip install --ignore-installed aiogram

pip install --ignore-installed tensorflow-gpu (or tensorflow-cpu)

pip install --ignore-installed opencv-python

pip install --ignore-installed sklearn


Done!

For launch the bot you can create .bat file (for Windows)

.bat file content:

@echo off

call %~dp0(your venv name)\Scripts\activate

set TOKEN=(your bot token)

python bot-main.py

pause
