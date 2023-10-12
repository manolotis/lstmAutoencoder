BATCH_SIZE=64
N_JOBS=8
BASE_SCRIPT="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/predict.py"
OUT_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/predictions/"


############# Trained and tested on clean data ############
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128

############# Trained on clean data, tested on perturbed data (no past) ############
# same test data path, different config
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing"
CONFIG = "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"

echo $CONFIG

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128 \
  --model-name-addition "no_past"
