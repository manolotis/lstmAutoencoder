BATCH_SIZE=8
N_JOBS=2
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
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128 \
  --model-name-addition "no_past"

############# Trained on clean data, tested on perturbed data (noisy heading) ############
# different test data and config
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing_noisy_heading"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_noisy_heading.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128 \
  --model-name-addition "noisy_heading"



############# Trained on clean and perturbed data, tested on perturbed data (no past) ############
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_no_past.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128_no_past__21445f6 --num-layers 3 --hidden-size 128 \
  --model-name-addition "retrained"

############# Trained on clean and perturbed data, tested on perturbed data (noisy heading) ############
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing_noisy_heading"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict_noisy_heading.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128_noisy_heading__21445f6 --num-layers 3 --hidden-size 128 \
  --model-name-addition "retrained"


############# Trained on clean and perturbed data, tested on clean data ############
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128_no_past__21445f6 --num-layers 3 --hidden-size 128 \
  --model-name-addition "retrained_unperturbed"

############# Trained on clean and perturbed data, tested on clean data ############
TEST_DATA_PATH="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing"
CONFIG="/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/predict.yaml"

python $BASE_SCRIPT --config $CONFIG --test-data-path "$TEST_DATA_PATH" --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128_noisy_heading__21445f6 --num-layers 3 --hidden-size 128 \
  --model-name-addition "retrained_unperturbed"