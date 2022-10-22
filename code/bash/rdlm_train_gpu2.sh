CUDA_VISIBLE_DEVICES=2 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_1x128.yaml" \

CUDA_VISIBLE_DEVICES=2 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_2x64.yaml" \

CUDA_VISIBLE_DEVICES=2 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/validation" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/train_3x512.yaml" \
