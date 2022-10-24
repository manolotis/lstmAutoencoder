TEST_DATA_PATH="/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/testing"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/predictions/"
BATCH_SIZE=128
N_JOBS=48
BASE_CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/predict.py"

python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_1x64__0303335 --num-layers 1 --hidden-size 64
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_1x128__0303335 --num-layers 1 --hidden-size 128
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_1x256__0303335 --num-layers 1 --hidden-size 256
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_1x512__0303335 --num-layers 1 --hidden-size 512

python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_2x64__0303335 --num-layers 2 --hidden-size 64
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_2x128__0303335 --num-layers 2 --hidden-size 128
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_2x256__0303335 --num-layers 2 --hidden-size 256
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_2x512__0303335 --num-layers 2 --hidden-size 512

python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x64__0303335 --num-layers 3 --hidden-size 64
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x256__0303335 --num-layers 3 --hidden-size 256
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x512__0303335 --num-layers 3 --hidden-size 512
