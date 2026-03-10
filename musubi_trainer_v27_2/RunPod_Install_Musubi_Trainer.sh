
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

cd /workspace

git clone --depth 1 https://github.com/FurkanGozukara/SECourses_Musubi_Trainer

cd SECourses_Musubi_Trainer

git reset --hard

git pull

git clone --depth 1 https://github.com/FurkanGozukara/convert_to_quant

cd convert_to_quant

git reset --hard

git pull

cd ..

git clone --depth 1 https://github.com/kohya-ss/musubi-tuner

cd musubi-tuner

git reset --hard

git pull

cd ..

# Check if Python 3.10 is available, if not install it
if ! command -v python3.10 &> /dev/null
then
    echo "Python 3.10 not found, installing..."
    apt-get update
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y python3.10 python3.10-venv python3.10-dev
    echo "Python 3.10 installed successfully"
else
    echo "Python 3.10 is already installed"
fi

# Create venv with Python 3.10 if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python 3.10 virtual environment..."
    python3.10 -m venv venv
else
    echo "Virtual environment already exists"
fi

source venv/bin/activate

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_musubi_trainer.txt

cd SECourses_Musubi_Trainer

cd musubi-tuner

uv pip install -e .

cd ..

cd convert_to_quant

uv pip install -e .

echo installation completed check for errors

unset LD_LIBRARY_PATH

cd ..

python gui.py --share