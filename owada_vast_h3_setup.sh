#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Owada Vast H3 Setup v2.2
# Vast.ai + RTX 5090 + ComfyUI + MiniMax H3 Ref2VA Turbo
#
# v2.2:
#   - Vast provisioning-safe /venv PATH
#   - MiniMax H3 Ref2VA automatic setup
#   - RunPod S3 workflow restore
#   - Automatic MP4 uploader to RunPod S3
#   - Individual "aws s3 cp" uploads (no ListObjectsV2 required)
#   - Existing models are skipped
#   - S3 failure does not break core H3 setup
# ============================================================

START_TIME=$(date +%s)

# Vast On-start runs before the interactive shell activates /venv/main.
export PATH="/venv/main/bin:${PATH}"

COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"

MODEL_REPO="Comfy-Org/MiniMax-H3"
TURBO_REPO="lightx2v/Minimax-h3-Turbo"
WORKFLOW_BASE="https://raw.githubusercontent.com/ModelTC/Minimax-H3-Turbo/main"

DIFFUSION_FILE="minimax_h3_ref2va_pruned_int8_convrot.safetensors"
TEXT_ENCODER_FILE="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_FILE="minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE_FILE="minimax_h3_audio_vae_fp32.safetensors"
LORA_FILE="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"

WORKFLOW_FILE="video_minimax_h3_ref2v_lightx2v_turbo.json"
WORKFLOW_URL="${WORKFLOW_BASE}/example_workflows/${WORKFLOW_FILE}"

DIFFUSION_DIR="${COMFY_DIR}/models/diffusion_models"
TEXT_ENCODER_DIR="${COMFY_DIR}/models/text_encoders"
VAE_DIR="${COMFY_DIR}/models/vae"
LORA_DIR="${COMFY_DIR}/models/loras"
WORKFLOW_DIR="${COMFY_DIR}/user/default/workflows"
OUTPUT_DIR="${COMFY_DIR}/output"

UPLOADER_SCRIPT="/workspace/owada_s3_output_uploader.sh"
UPLOADER_LOG="/workspace/owada_s3_output_uploader.log"
UPLOADER_PID="/workspace/owada_s3_output_uploader.pid"
UPLOADER_STATE="/workspace/.owada_s3_uploaded"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

log() {
    echo
    echo "[$(timestamp)] $*"
}

ok() {
    echo "[OK] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

trap 'echo; echo "[ERROR] Setup failed at line ${LINENO}."; exit 1' ERR

echo
echo "============================================================"
echo " Owada Vast H3 Setup v2.2"
echo "============================================================"
echo

# ------------------------------------------------------------
# 1. Wait for ComfyUI
# ------------------------------------------------------------

log "Waiting for ComfyUI directory..."

for i in {1..120}; do
    if [[ -d "${COMFY_DIR}" ]]; then
        break
    fi
    sleep 5
done

[[ -d "${COMFY_DIR}" ]] || \
    die "ComfyUI directory did not become available."

ok "ComfyUI found: ${COMFY_DIR}"

# ------------------------------------------------------------
# 2. GPU
# ------------------------------------------------------------

log "Checking GPU..."

command -v nvidia-smi >/dev/null 2>&1 || \
    die "nvidia-smi not found."

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
GPU_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"

echo "GPU : ${GPU_NAME}"
echo "VRAM: ${GPU_VRAM}"

if [[ "${GPU_NAME}" != *"RTX 5090"* ]]; then
    warn "Optimized for RTX 5090. Detected: ${GPU_NAME}"
fi

nvidia-smi || true

# ------------------------------------------------------------
# 3. Disk
# ------------------------------------------------------------

log "Checking disk..."

df -h /workspace

AVAILABLE_KB="$(df --output=avail /workspace | tail -1 | tr -d ' ')"
MIN_FREE_KB=$((70 * 1024 * 1024))

if (( AVAILABLE_KB < MIN_FREE_KB )); then
    die "Less than 70 GB free in /workspace."
fi

ok "Disk capacity OK."

# ------------------------------------------------------------
# 4. Network
# ------------------------------------------------------------

log "Checking Hugging Face..."

curl -fsSIL \
    --connect-timeout 10 \
    --max-time 20 \
    https://huggingface.co/ \
    >/dev/null || \
    die "Hugging Face unreachable."

ok "Hugging Face reachable."

log "Checking GitHub..."

curl -fsSIL \
    --connect-timeout 10 \
    --max-time 20 \
    https://github.com/ \
    >/dev/null || \
    die "GitHub unreachable."

ok "GitHub reachable."

# ------------------------------------------------------------
# 5. Hugging Face CLI
# ------------------------------------------------------------

log "Checking Hugging Face CLI..."

if ! command -v hf >/dev/null 2>&1; then
    /venv/main/bin/python -m pip install -U "huggingface_hub[cli]"
fi

command -v hf >/dev/null 2>&1 || \
    die "hf CLI unavailable."

ok "hf CLI available."

if [[ -n "${HF_TOKEN:-}" ]]; then
    ok "HF_TOKEN detected."
else
    warn "HF_TOKEN not set. Using unauthenticated public downloads."
fi

# ------------------------------------------------------------
# 6. Directories
# ------------------------------------------------------------

mkdir -p \
    "${DIFFUSION_DIR}" \
    "${TEXT_ENCODER_DIR}" \
    "${VAE_DIR}" \
    "${LORA_DIR}" \
    "${WORKFLOW_DIR}" \
    "${OUTPUT_DIR}"

# ------------------------------------------------------------
# 7. Hugging Face download helper
# ------------------------------------------------------------

hf_download_file() {
    local repo="$1"
    local remote_path="$2"
    local dest_dir="$3"
    local filename
    local temp_dir
    local downloaded
    local file_start
    local file_end

    filename="$(basename "${remote_path}")"

    if [[ -s "${dest_dir}/${filename}" ]]; then
        ok "Already exists: ${filename}"
        return
    fi

    log "Downloading ${filename}..."

    file_start=$(date +%s)

    temp_dir="$(mktemp -d /workspace/hf-download-XXXXXX)"

    if [[ -n "${HF_TOKEN:-}" ]]; then
        hf download \
            "${repo}" \
            "${remote_path}" \
            --local-dir "${temp_dir}" \
            --token "${HF_TOKEN}"
    else
        hf download \
            "${repo}" \
            "${remote_path}" \
            --local-dir "${temp_dir}"
    fi

    downloaded="${temp_dir}/${remote_path}"

    [[ -s "${downloaded}" ]] || \
        die "Downloaded file not found: ${downloaded}"

    mv -f "${downloaded}" "${dest_dir}/${filename}"
    rm -rf "${temp_dir}"

    [[ -s "${dest_dir}/${filename}" ]] || \
        die "Installation failed: ${filename}"

    file_end=$(date +%s)

    ok "Installed ${filename} in $((file_end - file_start)) sec"
}

# ------------------------------------------------------------
# 8. MiniMax H3 models
# ------------------------------------------------------------

hf_download_file \
    "${MODEL_REPO}" \
    "diffusion_models/${DIFFUSION_FILE}" \
    "${DIFFUSION_DIR}"

hf_download_file \
    "${MODEL_REPO}" \
    "text_encoders/${TEXT_ENCODER_FILE}" \
    "${TEXT_ENCODER_DIR}"

hf_download_file \
    "${MODEL_REPO}" \
    "vae/${VIDEO_VAE_FILE}" \
    "${VAE_DIR}"

hf_download_file \
    "${MODEL_REPO}" \
    "vae/${AUDIO_VAE_FILE}" \
    "${VAE_DIR}"

hf_download_file \
    "${TURBO_REPO}" \
    "${LORA_FILE}" \
    "${LORA_DIR}"

# ------------------------------------------------------------
# 9. Official fallback workflow
# ------------------------------------------------------------

log "Installing official Ref2VA workflow..."

if [[ ! -s "${WORKFLOW_DIR}/${WORKFLOW_FILE}" ]]; then
    curl -fL \
        --retry 3 \
        --retry-delay 3 \
        --connect-timeout 15 \
        "${WORKFLOW_URL}" \
        -o "${WORKFLOW_DIR}/${WORKFLOW_FILE}"
fi

[[ -s "${WORKFLOW_DIR}/${WORKFLOW_FILE}" ]] || \
    die "Workflow installation failed."

ok "Official workflow installed."

# ------------------------------------------------------------
# 10. RunPod S3 configuration
# ------------------------------------------------------------

log "Checking RunPod S3 configuration..."

S3_READY=1

REQUIRED_S3_VARS=(
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION
    RUNPOD_S3_ENDPOINT
    RUNPOD_S3_BUCKET
)

for var in "${REQUIRED_S3_VARS[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        warn "${var} is not set."
        S3_READY=0
    fi
done

# ------------------------------------------------------------
# 11. AWS CLI
# ------------------------------------------------------------

if (( S3_READY == 1 )); then

    log "Checking AWS CLI..."

    if ! command -v aws >/dev/null 2>&1; then
        /venv/main/bin/python -m pip install awscli
    fi

    if command -v aws >/dev/null 2>&1; then
        ok "AWS CLI available."
    else
        warn "AWS CLI installation failed."
        S3_READY=0
    fi

else
    warn "RunPod S3 configuration incomplete."
fi

# ------------------------------------------------------------
# 12. Restore workflows from RunPod S3
# ------------------------------------------------------------

if (( S3_READY == 1 )); then

    log "Restoring workflows from RunPod S3..."

    if aws s3 sync \
        "s3://${RUNPOD_S3_BUCKET}/vast/workflows/" \
        "${WORKFLOW_DIR}/" \
        --endpoint-url "${RUNPOD_S3_ENDPOINT}" \
        --region "${AWS_DEFAULT_REGION}"
    then
        ok "RunPod S3 workflows restored."
    else
        warn "RunPod S3 workflow sync failed."
        warn "Core H3 setup will continue."
    fi

else
    warn "Skipping RunPod S3 workflow restore."
fi

# ------------------------------------------------------------
# 13. Validate core H3 installation
# ------------------------------------------------------------

log "Validating core H3 installation..."

REQUIRED_FILES=(
    "${DIFFUSION_DIR}/${DIFFUSION_FILE}"
    "${TEXT_ENCODER_DIR}/${TEXT_ENCODER_FILE}"
    "${VAE_DIR}/${VIDEO_VAE_FILE}"
    "${VAE_DIR}/${AUDIO_VAE_FILE}"
    "${LORA_DIR}/${LORA_FILE}"
    "${WORKFLOW_DIR}/${WORKFLOW_FILE}"
)

for file in "${REQUIRED_FILES[@]}"; do
    [[ -s "${file}" ]] || die "Missing: ${file}"
    echo "[OK] $(du -h "${file}" | cut -f1)  ${file}"
done

# ------------------------------------------------------------
# 14. Create automatic MP4 uploader
# ------------------------------------------------------------

if (( S3_READY == 1 )); then

    log "Creating automatic MP4 uploader..."

    cat > "${UPLOADER_SCRIPT}" <<'UPLOADER_EOF'
#!/usr/bin/env bash
set -u

SOURCE_DIR="/workspace/ComfyUI/output"
DEST_PREFIX="s3://${RUNPOD_S3_BUCKET}/ComfyUI/output/video"
STATE_DIR="/workspace/.owada_s3_uploaded"
INTERVAL=5

mkdir -p "${SOURCE_DIR}" "${STATE_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Owada Vast -> RunPod S3 uploader started"
log "Source: ${SOURCE_DIR}"
log "Dest:   ${DEST_PREFIX}"
log "Mode:   individual aws s3 cp"
log "Interval: ${INTERVAL}s"

while true; do

    while IFS= read -r -d '' file; do

        filename="$(basename "${file}")"
        s3_filename="${filename%.mp4}-audio.mp4"
        marker="${STATE_DIR}/${filename}.done"

        # Already uploaded during this Vast instance.
        [[ -f "${marker}" ]] && continue

        # Do not upload an MP4 while ComfyUI may still be writing it.
        now=$(date +%s)
        mtime=$(stat -c %Y "${file}" 2>/dev/null || echo "${now}")
        age=$((now - mtime))

        (( age >= 5 )) || continue

        log "Uploading: ${filename}"

        if aws s3 cp \
            "${file}" \
            "${DEST_PREFIX}/${s3_filename}"
            --endpoint-url "${RUNPOD_S3_ENDPOINT}" \
            --region "${AWS_DEFAULT_REGION}" \
            --only-show-errors
        then
            touch "${marker}"
            log "Uploaded: ${filename}"
        else
            log "WARN: Upload failed: ${filename}"
        fi

    done < <(
        find "${SOURCE_DIR}" \
            -type f \
            -iname '*.mp4' \
            -print0 2>/dev/null
    )

    sleep "${INTERVAL}"
done
UPLOADER_EOF

    chmod +x "${UPLOADER_SCRIPT}"

    ok "Automatic MP4 uploader created."

    # --------------------------------------------------------
    # Stop an old uploader if setup.sh is executed again
    # --------------------------------------------------------

    if [[ -f "${UPLOADER_PID}" ]]; then

        OLD_PID="$(cat "${UPLOADER_PID}" 2>/dev/null || true)"

        if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
            log "Stopping previous MP4 uploader (PID ${OLD_PID})..."
            kill "${OLD_PID}" 2>/dev/null || true
            sleep 1
        fi

    fi

    # --------------------------------------------------------
    # Start uploader
    # --------------------------------------------------------

    log "Starting automatic MP4 uploader..."

    nohup "${UPLOADER_SCRIPT}" \
        > "${UPLOADER_LOG}" 2>&1 &

    NEW_PID=$!

    echo "${NEW_PID}" > "${UPLOADER_PID}"

    sleep 1

    if kill -0 "${NEW_PID}" 2>/dev/null; then
        ok "MP4 uploader running. PID: ${NEW_PID}"
    else
        warn "MP4 uploader failed to start."
    fi

else
    warn "Skipping automatic MP4 uploader."
fi

# ------------------------------------------------------------
# 15. Final status
# ------------------------------------------------------------

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo
echo "============================================================"
echo " H3 REF2VA READY"
echo "============================================================"
echo
echo "GPU:"
echo "  ${GPU_NAME}"
echo
echo "Setup time:"
echo "  ${TOTAL_TIME} seconds"
echo "  $((TOTAL_TIME / 60))m $((TOTAL_TIME % 60))s"
echo
echo "Workflows:"
find "${WORKFLOW_DIR}" \
    -maxdepth 1 \
    -type f \
    -name '*.json' \
    -printf '  %f\n' 2>/dev/null || true

if (( S3_READY == 1 )); then
    echo
    echo "RunPod S3:"
    echo "  Workflow restore : ENABLED"
    echo "  MP4 auto upload  : ENABLED"
    echo "  Output target    : s3://${RUNPOD_S3_BUCKET}/ComfyUI/output/"
    echo
    echo "Uploader log:"
    echo "  ${UPLOADER_LOG}"
fi

echo
echo "Disk:"
df -h /workspace

echo
echo "============================================================"
