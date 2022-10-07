CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/prerender/prerender.py \
--data-path "/media/disk1/datasets/waymo/motion v1.0/uncompressed/tf_example/training/" \
--output-path "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/data/prerendered/training" \
--config "/home/manolotis/sandbox/robustness_benchmark/lstmAutoencoder/code/configs/prerender.yaml" \
--n-jobs 48