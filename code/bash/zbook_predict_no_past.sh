TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/validation"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/predictions/"
BATCH_SIZE=8
N_JOBS=2
BASE_CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/predict.py"

python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path $TEST_DATA_PATH --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_1x128__0303335 --num-layers 1 --hidden-size 128 \
  --model-name-addition "no_past"
python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path $TEST_DATA_PATH --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128 \
  --model-name-addition "no_past"