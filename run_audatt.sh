# # # #!/bin/bash

# # Common configs
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# AUDIO_ENCODER="Beats"
# LEARNED_EMBEDS="output_train/learned_embeds.bin"
# SHOW_ATTN="No"

# # Pipelines to iterate
# PIPELINES=("AT" "mixed")

# # Dataset configs
# DATA_NAMES=("VGGi")
# DATA_DIRS=(
#     "../dataset/VGG_Sound_extracted/test_data_prog/"
# )
# OUTPUT_DIRS=(
#     "output_vggi/"
# )

# # Loop over pipelines and datasets
# for PIPELINE in "${PIPELINES[@]}"; do
#   for i in ${!DATA_NAMES[@]}; do
    
#     RUN_NAME="${DATA_NAMES[$i]}_${PIPELINE}"
#     DATA_DIR="${DATA_DIRS[$i]}"
#     OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

#     echo "Running: $RUN_NAME"

#     python audatti.py \
#       --pretrained_model_name_or_path="$MODEL_NAME" \
#       --run_name="$RUN_NAME" \
#       --pipeline="$PIPELINE" \
#       --audio_encoder="$AUDIO_ENCODER" \
#       --data_dir="$DATA_DIR" \
#       --output_dir="$OUTPUT_DIR" \
#       --show_attn="$SHOW_ATTN" \
#       --learned_embeds="$LEARNED_EMBEDS"

#   done
# done



# # Common configs
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# AUDIO_ENCODER="Beats"
# LEARNED_EMBEDS="AudioToken/output/embedder_learned_embeds.bin"
# SHOW_ATTN="No"

# # Pipelines to iterate
# PIPELINES=("AT")

# # Dataset configs
# DATA_NAMES=("VGGi")
# DATA_DIRS=(
#     "../dataset/VGG_Sound_extracted/test_data_prog/"
# )
# OUTPUT_DIRS=(
#     "output_vggi/"
# )

# # Loop over pipelines and datasets
# for PIPELINE in "${PIPELINES[@]}"; do
#   for i in ${!DATA_NAMES[@]}; do
    
#     RUN_NAME="${DATA_NAMES[$i]}_${PIPELINE}"_org
#     DATA_DIR="${DATA_DIRS[$i]}"
#     OUTPUT_DIR="${OUTPUT_DIRS[$i]}"

#     echo "Running: $RUN_NAME"

#     python audatti.py \
#       --pretrained_model_name_or_path="$MODEL_NAME" \
#       --run_name="$RUN_NAME" \
#       --pipeline="$PIPELINE" \
#       --audio_encoder="$AUDIO_ENCODER" \
#       --data_dir="$DATA_DIR" \
#       --output_dir="$OUTPUT_DIR" \
#       --show_attn="$SHOW_ATTN" \
#       --learned_embeds="$LEARNED_EMBEDS"

#   done
# done



# # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export RUN_NAME="IS3_AT"
# # export DATA_DIR="../dataset/IS3/"
# # export OUTPUT_DIR="output_is3/"
# # export GENERATED_IMGS_DIR="output_is3/imgs/IS3_AT/"
# # python eval_metrics.py   --pretrained_model_name_or_path="$MODEL_NAME"  --run_name="$RUN_NAME"  --data_dir="$DATA_DIR"  --output_dir="$OUTPUT_DIR"  --generated_imgs_dir="$GENERATED_IMGS_DIR"


# # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export RUN_NAME="VGG_Beats"
# # export PIPELINE="AT"
# # export AUDIO_ENCODER="Beats"
# # export DATA_DIR="../dataset/VGG_Sound_extracted/test_data"
# # export OUTPUT_DIR="output_vgg/"
# # export LEARNED_EMBEDS="output_train/learned_embeds.bin"
# # python plot_clusters.py   --pretrained_model_name_or_path="$MODEL_NAME"  --run_name="$RUN_NAME"  --use_label="$USE_LABEL"  --pipeline="$PIPELINE"  --audio_encoder="$AUDIO_ENCODER"  --data_dir="$DATA_DIR"  --output_dir="$OUTPUT_DIR"   --learned_embeds="$LEARNED_EMBEDS"

# # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export RUN_NAME="VGG_Clap"
# # export PIPELINE="AT"
# # export AUDIO_ENCODER="Clap"
# # export DATA_DIR="../dataset/VGG_Sound_extracted/test_data"
# # export OUTPUT_DIR="output_vgg/"
# # export LEARNED_EMBEDS="output_train_claps/learned_embeds.bin"
# # python plot_clusters.py   --pretrained_model_name_or_path="$MODEL_NAME"  --run_name="$RUN_NAME"  --use_label="$USE_LABEL"  --pipeline="$PIPELINE"  --audio_encoder="$AUDIO_ENCODER"  --data_dir="$DATA_DIR"  --output_dir="$OUTPUT_DIR"   --learned_embeds="$LEARNED_EMBEDS"

# # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export RUN_NAME=""
# # export DATA_DIR="../dataset/VGG_Sound_extracted/temp_data"
# # export OUTPUT_DIR="output_s2s/"
# # export GENERATED_IMGS_DIR="output_s2s/imgs/"
# # python eval_metrics.py   --pretrained_model_name_or_path="$MODEL_NAME"  --run_name="$RUN_NAME"  --data_dir="$DATA_DIR"  --output_dir="$OUTPUT_DIR"  --generated_imgs_dir="$GENERATED_IMGS_DIR"

# # export MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # export RUN_NAME=""
# # export DATA_DIR="../dataset/VGG_Sound_extracted/test_data"
# # export OUTPUT_DIR="output_sd/"
# # export GENERATED_IMGS_DIR="output_sd/imgs/"
# # python eval_metrics.py   --pretrained_model_name_or_path="$MODEL_NAME"  --run_name="$RUN_NAME"  --data_dir="$DATA_DIR"  --output_dir="$OUTPUT_DIR"  --generated_imgs_dir="$GENERATED_IMGS_DIR"


#!/bin/bash

# Common configs
MODEL_NAME="CompVis/stable-diffusion-v1-4"
AUDIO_ENCODER="Clap"
LEARNED_EMBEDS="output_train_claps/learned_embeds.bin"
SHOW_ATTN="No"

# Pipelines to iterate
PIPELINES=("ane")

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
