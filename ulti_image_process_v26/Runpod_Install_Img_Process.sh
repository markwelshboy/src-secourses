
apt-get update --yes
apt-get install python3-tk --yes

python -m venv venv

source ./venv/bin/activate

python -m pip install --upgrade pip

echo "Installing requirements"

pip install -r requirements_image_process.txt

git clone https://github.com/IDEA-Research/GroundingDINO

python HF_model_downloader_img_process.py

# Show completion message
echo "Virtual environment made and installed properly"

