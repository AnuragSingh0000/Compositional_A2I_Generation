
#!/bin/bash

# Common configs
MODEL_NAME="CompVis/stable-diffusion-v1-4"
AUDIO_ENCODER="Beats"
LEARNED_EMBEDS="output_train/learned_embeds.bin"
SHOW_ATTN="No"

# Pipelines to iterate
PIPELINES=("AT" "mixed")

# Dataset configs
DATA_NAMES=("VGGN")
DATA_DIRS=(
    "../dataset/VGG_Sound_extracted/test_data_group"
)
OUTPUT_DIRS=(
    "output_time/vggn/"
)

# Loop over pipelines and datasets
for PIPELINE in "${PIPELINES[@]}"; do
  for i in "${!DATA_NAMES[@]}"; do

    RUN_NAME="${PIPELINE}"
    DATA_DIR="${DATA_DIRS[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

    echo "Running: $RUN_NAME on ${DATA_NAMES[$i]}"

    python audtime.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --run_name="$RUN_NAME" \
      --pipeline="$PIPELINE" \
      --audio_encoder="$AUDIO_ENCODER" \
      --data_dir="$DATA_DIR" \
      --output_dir="$OUTPUT_DIR" \
      --show_attn="$SHOW_ATTN" \
      --learned_embeds="$LEARNED_EMBEDS"

  done
done