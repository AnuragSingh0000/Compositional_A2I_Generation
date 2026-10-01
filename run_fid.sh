FID_SCRIPT="get_fid.py"
REAL_FOLDER="../dataset/VGG_Sound_extracted/test_data/"
N=1
run_name="VGG_macs"
folder="output_macs/VGG/imgs/"


echo "Running FID for: $run_name"

python "$FID_SCRIPT" \
            --n $N \
            --gen_folder "$folder" \
            --real_folder "$REAL_FOLDER" \
            --run_name "$run_name"

echo "All runs completed!"


FID_SCRIPT="get_fid.py"
REAL_FOLDER="../dataset/VGG_Sound_extracted/test_data_paired/"
N=2
run_name="VGG2_macs"
folder="output_macs/VGG2/imgs/"


echo "Running FID for: $run_name"

python "$FID_SCRIPT" \
            --n $N \
            --gen_folder "$folder" \
            --real_folder "$REAL_FOLDER" \
            --run_name "$run_name"

echo "All runs completed!"



FID_SCRIPT="get_fid.py"
REAL_FOLDER="../dataset/VGG_Sound_extracted/test_data_triples/"
N=3
run_name="VGG3_macs"
folder="output_macs/VGG3/imgs/"


echo "Running FID for: $run_name"

python "$FID_SCRIPT" \
            --n $N \
            --gen_folder "$folder" \
            --real_folder "$REAL_FOLDER" \
            --run_name "$run_name"

echo "All runs completed!"



FID_SCRIPT="get_fid.py"
REAL_FOLDER="../dataset/VGG_Sound_extracted/test_data_group/"
N=variable
run_name="VGGn_macs"
folder="output_macs/VGGn/imgs/"


echo "Running FID for: $run_name"

python "$FID_SCRIPT" \
            --n $N \
            --gen_folder "$folder" \
            --real_folder "$REAL_FOLDER" \
            --run_name "$run_name"

echo "All runs completed!"


FID_SCRIPT="get_fid.py"
REAL_FOLDER="../dataset/IS3/"
N=2
run_name="IS3_macs"
folder="output_macs/IS3/imgs/"


echo "Running FID for: $run_name"

python "$FID_SCRIPT" \
            --n $N \
            --gen_folder "$folder" \
            --real_folder "$REAL_FOLDER" \
            --run_name "$run_name"

echo "All runs completed!"




# FID_SCRIPT="get_fid.py"
# REAL_FOLDER="../dataset/VGG_Sound_extracted/test_data/"
# N=1
# run_name="VGG_AT"
# folder="output_vgg/imgs/VGG_AT"


# echo "Running FID for: $run_name"

# python "$FID_SCRIPT" \
#             --n $N \
#             --gen_folder "$folder" \
#             --real_folder "$REAL_FOLDER" \
#             --run_name "$run_name"





# echo "All runs completed!"