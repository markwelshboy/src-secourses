cd SwarmUI

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
set NVIDIA_TF32_OVERRIDE=1
set CUDA_MODULE_LOADING=LAZY

launch-windows.bat --port 7861