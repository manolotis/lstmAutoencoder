BATCH_SIZE=128
N_JOBS=48
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/predictions/"


############# Trained on clean data, tested on perturbed data (no past) ############
# same test data path, different config
TEST_DATA_PATH="/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128 \
  --model-name-addition "no_past"


############# Trained on clean and perturbed data, tested on perturbed data (no past) ############
TEST_DATA_PATH="/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128_no_past__21445f6 --num-layers 3 --hidden-size 128 \
  --model-name-addition "retrained"



