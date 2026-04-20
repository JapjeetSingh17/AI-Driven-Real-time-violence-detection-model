from huggingface_hub import hf_hub_download
import os

# Download pretrained violence detection model
# This is trained on RWF-2000 (fight detection dataset)
path = hf_hub_download(
    repo_id="dima806/violent_video_detection",
    filename="model.safetensors",
    local_dir="./weights"
)
print(f"Downloaded to: {path}")