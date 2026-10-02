#!/bin/bash
set -e


# -------- Sonic Diffusion --------
echo "Starting Sonic Diffusion inference..."

# for DIR in \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_1" \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_2" \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_3" \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_4" \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_5" \
#   "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav_6"
# do
#   echo "Running inference on directory: $DIR"
#   python inference.py "$DIR"
#   echo "Run completed."
# done


python inference.py "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav"

echo "All inference runs for Sonic Diffusion completed."
