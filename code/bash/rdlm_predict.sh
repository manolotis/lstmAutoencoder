TEST_DATA_PATH="/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmEncoderDecoder/testing"
OUT_PATH="/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/predictions/"
BATCH_SIZE=128
N_JOBS=48
BASE_CONFIG="/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/code/configs/predict.yaml"
BASE_SCRIPT="/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/code/predict.py"

python $BASE_SCRIPT --config $BASE_CONFIG --test-data-path $TEST_DATA_PATH --batch-size $BATCH_SIZE --n-jobs $N_JOBS --out-path $OUT_PATH \
  --model-name lstm_3x128__0303335 --num-layers 3 --hidden-size 128






