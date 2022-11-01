python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/train.py \
  --train-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/training" \
  --val-data-path "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/validation" \
  --train-data-path-noisy "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/training_noisy_heading" \
  --val-data-path-noisy "/home/manolotis/datasets/waymo/motion v1.0/prerender/lstmAutoencoder/validation_noisy_heading" \
  --config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/training/train_3x128_noisy_heading.yaml" \
