CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/prerender/prerender.py \
--data-path "/home/manolotis/sandbox/waymoMotion/data/reduced/tf_example/validation/" \
--output-path "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/testing_noisy_heading/" \
--config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/prerender.yaml" \
--n-jobs 8