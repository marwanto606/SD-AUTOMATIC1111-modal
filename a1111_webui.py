import os
import subprocess
import modal

PORT = 8000

# Buat volume persisten yang akan menyimpan file instalasi agar tidak perlu di-download ulang
vol = modal.Volume.from_name("a1111-cache", create_if_missing=True)

# Pada tahap build image, kita clone dan download ke folder /app/webui
a1111_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git",
        "aria2",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        # Clone kode ke folder /app/webui (bukan /webui)
        "git clone --depth 1 --branch v1.10.1 https://github.com/AUTOMATIC1111/stable-diffusion-webui /app/webui",
        "git clone --depth 1 https://github.com/BlafKing/sd-civitai-browser-plus /app/webui/extensions/sd-civitai-browser-plus",
        "git clone --depth 1 https://huggingface.co/embed/negative /app/webui/embeddings/negative",
        "git clone --depth 1 https://huggingface.co/embed/lora /app/webui/models/Lora/positive",
        "git clone --depth 1 https://github.com/camenduru/stable-diffusion-webui-images-browser /app/webui/extensions/stable-diffusion-webui-images-browser",
        # Download model ke folder yang sudah ditentukan dalam image
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/embed/upscale/resolve/main/4x-UltraSharp.pth -d /app/webui/models/ESRGAN -o 4x-UltraSharp.pth",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/artmozai/duchaiten-aiart-xl/resolve/main/duchaitenAiartSDXL_v33515.safetensors -d /app/webui/models/Stable-diffusion -o duchaitenAiartSDXL_v33515.safetensors",
        "aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/sdxl_vae/resolve/main/sdxl_vae.safetensors -d /app/webui/models/VAE -o sdxl_vae.safetensors",
        "python -m venv /app/webui/venv",
        "cd /app/webui && . venv/bin/activate && " +
        "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a100",
    )
    .run_commands(
        "cd /app/webui && . venv/bin/activate && " +
        "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
        gpu="a100",
    )
)

app = modal.App("a1111-webui", image=a1111_image)

# Mount volume persisten ke path /webui agar file instalasi tersimpan antar eksekusi.
@app.function(
    gpu="a100",
    cpu=2,
    memory=1024,
    timeout=3600,
    allow_concurrent_inputs=100,
    keep_warm=1,
    volumes={"/webui": vol}
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    # Jika folder /webui (yang dipersist) masih kosong, salin dari /app/webui (baked ke image)
    if not os.path.exists("/webui/launch.py"):
        subprocess.run("cp -r /app/webui/* /webui/", shell=True, check=True)
    
    # Ubah file shared_options.py dengan menambahkan opsi "sd_vae" dan "CLIP_stop_at_last_layers"
    os.system(
        r"sed -i -e 's/\[\"sd_model_checkpoint\"\]/\[\"sd_model_checkpoint\",\"sd_vae\",\"CLIP_stop_at_last_layers\"\]/g' /webui/modules/shared_options.py"
    )
    
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    launch.py \
        --skip-prepare-environment \
        --no-gradio-queue \
        --listen \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)
