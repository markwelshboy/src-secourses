cd /workspace

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

FFMPEG_URL="https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz"
FFMPEG_ARCHIVE="/tmp/ffmpeg-n8.0-latest-linux64-gpl-8.0.tar.xz"
FFMPEG_INSTALL_DIR="/opt/ffmpeg-n8.0"
FFMPEG_TMP_DIR="/tmp/ffmpeg-install-$$"

echo Installing ffmpeg from: "$FFMPEG_URL"
curl -L "$FFMPEG_URL" -o "$FFMPEG_ARCHIVE"
mkdir -p "$FFMPEG_TMP_DIR"
tar -xJf "$FFMPEG_ARCHIVE" -C "$FFMPEG_TMP_DIR"
FFMPEG_EXTRACTED_DIR="$(find "$FFMPEG_TMP_DIR" -maxdepth 1 -type d -name 'ffmpeg-n8.0-latest-linux64-gpl-8.0' | head -n 1)"
if [ -z "$FFMPEG_EXTRACTED_DIR" ]; then
  echo "ffmpeg extraction failed"
  exit 1
fi
rm -rf "$FFMPEG_INSTALL_DIR"
mv "$FFMPEG_EXTRACTED_DIR" "$FFMPEG_INSTALL_DIR"
rm -rf "$FFMPEG_TMP_DIR"
ln -sf "$FFMPEG_INSTALL_DIR/bin/ffmpeg" /usr/local/bin/ffmpeg
ln -sf "$FFMPEG_INSTALL_DIR/bin/ffprobe" /usr/local/bin/ffprobe
ln -sf "$FFMPEG_INSTALL_DIR/bin/ffplay" /usr/local/bin/ffplay
cat > /etc/profile.d/ffmpeg.sh <<'EOF'
export PATH="/opt/ffmpeg-n8.0/bin:$PATH"
EOF
chmod 644 /etc/profile.d/ffmpeg.sh
hash -r || true
ffmpeg -version | head -n 1

git clone https://github.com/FurkanGozukara/SECourses_Premium_Upscaler_Pro

cd SECourses_Premium_Upscaler_Pro

python3.10 -m venv venv

source venv/bin/activate

python3 -m pip install --upgrade pip

pip install uv

cd ..

uv pip install -r requirements.txt

cd SECourses_Premium_Upscaler_Pro

git clone https://github.com/FurkanGozukara/RIFE_Ultimate_Video_Upscaler RIFE

git clone https://github.com/FurkanGozukara/Video_Comparison_Slider

git clone https://github.com/FurkanGozukara/SeedVR2

git clone https://github.com/furkanGozukara/FlashVSR_plus

cd SeedVR2

git reset --hard

git pull

cd ..

cd RIFE

git reset --hard

git pull

cd ..

cd FlashVSR_plus

git reset --hard

git pull

cd ..

cd ..

python3 Models_Downloader.py --all

echo .
echo .
echo .
echo .
echo Ultimate Video Image Upscaler installed check out all messages and save them before close to verify any errors or not later
