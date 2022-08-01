python -m venv venv

call %~dp0venv\Scripts\activate

pip install --ignore-installed aiogram

pip install --ignore-installed tensorflow-cpu

pip install --ignore-installed opencv-python

pip install --ignore-installed sklearn