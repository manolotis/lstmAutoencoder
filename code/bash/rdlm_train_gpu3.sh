CUDA_VISIBLE_DEVICES=3 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_1x256.yaml" \

CUDA_VISIBLE_DEVICES=3 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_2x128.yaml" \

CUDA_VISIBLE_DEVICES=3 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/multipathPP/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_3x64.yaml" \
