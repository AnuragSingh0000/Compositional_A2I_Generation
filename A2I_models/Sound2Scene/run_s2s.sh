# -------- Sound2Scene --------
cd ../Sound2Scene

echo "Starting Sound2Scene tests..."

OUT_PATH="../output_s2s"
CHECKPOINT="./checkpoints/sound2scene.pth"

for DIR in \
  "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_wav" 
do
  echo "Running Sound2Scene test on: $DIR"
  python test1.py \
    --checkpoint_vggish "$CHECKPOINT" \
    --wav_path "$DIR" \
    --out_path "$OUT_PATH"
  echo "Test completed for directory: $DIR"
done

echo "All Sound2Scene tests completed."
