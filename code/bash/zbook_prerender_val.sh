CUDA_VISIBLE_DEVICES=-1 \
python /home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/code/prerender/prerender.py \
--data-path "/home/manolotis/sandbox/datasets/waymo_v1.1/uncompressed/tf_example/validation/" \
--output-path "/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/data/prerendered/validation" \
--config "/home/manolotis/sandbox/scenario_based_evaluation/lstmEncoderDecoder/code/configs/prerender.yaml" \
--n-jobs 8