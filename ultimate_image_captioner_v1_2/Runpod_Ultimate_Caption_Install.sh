
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

git clone https://github.com/FurkanGozukara/Ultimate_Image_Captioner_Pro

cd Ultimate_Image_Captioner_Pro

git reset --hard

git pull

python3.11 -m venv venv

source ./venv/bin/activate

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_premium_caption.txt  --index-strategy unsafe-best-match
uv pip install xformers==0.0.35 --no-deps --index-url https://download.pytorch.org/whl/cu130

echo if any download error happens re run install script

python HF_model_downloader.py

