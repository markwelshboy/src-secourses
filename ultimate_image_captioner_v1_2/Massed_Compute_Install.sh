
export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

git clone https://github.com/FurkanGozukara/Ultimate_Image_Captioner_Pro

cd Ultimate_Image_Captioner_Pro

git reset --hard

git pull

# Remove existing Python 3.11 RC version
echo "Removing existing Python 3.11 (RC)..."
sudo apt-get remove -y python3.11 python3.11-venv python3.11-dev python3.11-distutils 2>/dev/null
sudo apt-get autoremove -y

# Add deadsnakes PPA
echo "Adding deadsnakes PPA..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

# Pin deadsnakes packages higher than the Ubuntu repo to prevent RC installs
sudo tee /etc/apt/preferences.d/deadsnakes-python3.11 > /dev/null <<EOF
Package: python3.11 python3.11-*
Pin: release o=LP-PPA-deadsnakes-ppa
Pin-Priority: 1000
EOF

# Show what will be installed
echo "Available Python 3.11 versions:"
apt-cache policy python3.11

# Install Python 3.11 from deadsnakes (candidate is already 3.11.15)
echo "Installing latest stable Python 3.11..."
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

python3.11 -m venv venv

source ./venv/bin/activate

python -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements_premium_caption.txt  --index-strategy unsafe-best-match
uv pip install xformers==0.0.35 --no-deps --index-url https://download.pytorch.org/whl/cu130

echo if any download error happens re run install script

python3 HF_model_downloader.py

