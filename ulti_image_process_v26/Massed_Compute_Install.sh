pip install requests
pip install tqdm
sudo apt update

python3 -m venv venv

source ./venv/bin/activate

python3 -m pip install --upgrade pip

echo "Installing requirements"

pip install -r requirements_image_process.txt

git clone https://github.com/IDEA-Research/GroundingDINO

python3 HF_model_downloader_img_process.py

# Keep the terminal open
read -p "Press Enter to continue..."