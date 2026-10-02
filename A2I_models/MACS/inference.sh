# export DATA_DIR='/root/autodl-tmp/datasets/'                    # will find data dir according to DATASET_NAME
# export OUTPUT_DIR="/root/autodl-tmp/output/proposed/inference/" # will find output dir according to DATASET_NAME
# export DATASET_NAME='Landscape'

# accelerate launch inference.py \
#   --data_dir=$DATA_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --dataset_name=$DATASET_NAME \
#   --M=6 --cfg \
#   --prompt '' \
#   --negative_prompt ''


# export DATA_DIR='/iitgn/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_triples/'                    
# export OUTPUT_DIR="../output_macs"
# export DATASET_NAME='VGG3'
# export WEIGHTS_NAME='AudioSet'

# accelerate launch inference_sep.py \
#   --data_dir=$DATA_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --dataset_name=$DATASET_NAME \
#   --weights_name=$WEIGHTS_NAME \
#   --M=6 --cfg \
#   --prompt '' \
#   --negative_prompt ''

export DATA_DIR='/iitgn/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data/'                    
export OUTPUT_DIR="../output_macs"
export DATASET_NAME='VGG'
export WEIGHTS_NAME='AudioSet'

accelerate launch inference_sep.py \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --weights_name=$WEIGHTS_NAME \
  --M=6 --cfg \
  --prompt '' \
  --negative_prompt ''

