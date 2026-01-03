
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

cd /workspace

git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer

cd SECourses_Musubi_Trainer

git reset --hard

git pull

git clone --depth 1 https://github.com/kohya-ss/musubi-tuner

cd musubi-tuner

git reset --hard

git pull

cd ..

python -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_trainer.txt

cd SECourses_Musubi_Trainer

cd musubi-tuner

uv pip install -e .

echo installation completed check for errors

unset LD_LIBRARY_PATH

cd ..

python gui.py --share