

git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
git clone --depth 1 https://github.com/welltop-cn/ComfyUI-TeaCache SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-TeaCache
git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux SwarmUI/src/BuiltinExtensions/ComfyUIBackend/DLNodes/comfyui_controlnet_aux

cd SwarmUI

git reset --hard

git pull

cd src/BuiltinExtensions/ComfyUIBackend/DLNodes/ComfyUI-Frame-Interpolation
git reset --hard

git pull

cd .. 
cd ComfyUI-TeaCache

git reset --hard

git pull

cd .. 
cd comfyui_controlnet_aux

git reset --hard

git pull

cd ..
cd ..
cd ..
cd ..
cd ..

echo SwarmUI updated to latest

launch-windows.bat --port 7861

pause