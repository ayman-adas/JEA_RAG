# vLLM Server Setup

Run these on the server. This creates a **separate** Python environment for vLLM — do not install it into the chatbot app's venv.

## 1. Check GPU is visible
```bash
nvidia-smi
```
If this errors, install NVIDIA drivers first (ask your hosting provider — GPU servers usually come with them preinstalled).

## 2. Create the venv
```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
```

## 3. Install vLLM
```bash
pip install vllm
```

## 4. Start the server
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B-AWQ \
  --served-model-name jea-chat \
  --port 8001 \
  --kv-cache-memory <BYTES> \
  --max-model-len 8192
```
Replace `<BYTES>` with how much RAM to give the KV cache, in bytes (e.g. `13016231424` = ~12GB).

Leave this running. First start takes a few minutes (downloads the model).

## 5. Keep it running after you close the terminal
```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null <<EOF
[Unit]
Description=vLLM Server
After=network.target

[Service]
User=$USER
ExecStart=$HOME/vllm-env/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B-AWQ --served-model-name jea-chat --port 8001 --kv-cache-memory 13016231424 --max-model-len 8192
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now vllm
```
Check it's up:
```bash
sudo systemctl status vllm
```

## 6. Test it
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "jea-chat", "messages": [{"role": "user", "content": "hello"}]}'
```
If you get a reply back, it's working.

The chatbot app already defaults to `localhost:8001` and model name `jea-chat`, matching this setup — no `.env` changes needed. Only add `VLLM_BASE_URL` / `VLLM_CHAT_MODEL` to the chatbot app's `.env` if vLLM ends up on a different machine or a different `--served-model-name`.
