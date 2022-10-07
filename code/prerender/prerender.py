import multiprocessing
from tqdm import tqdm
import tensorflow as tf
from utils.prerender_utils import get_renderers, create_dataset, parse_arguments, merge_and_save
from utils.utils import get_config
from utils.features_description import generate_features_description


def main():
    args = parse_arguments()
    config = get_config(args)
    dataset = create_dataset(config["data_path"], config["n_shards"], config["shard_id"])
    renderers = get_renderers(config["renderers"])

    if config["multiprocessing"]:

        p = multiprocessing.Pool(args.n_jobs)
        processes = []
        k = 0
        for data in tqdm(dataset.as_numpy_iterator()):
            k += 1
            data = tf.io.parse_single_example(data, generate_features_description())
            processes.append(
                p.apply_async(
                    merge_and_save,
                    kwds=dict(
                        renderers=renderers,
                        data=data,
                        output_path=args.output_path,
                    ),
                )
            )

        for r in tqdm(processes):
            r.get()
    else:
        for data in tqdm(dataset.as_numpy_iterator()):
            data = tf.io.parse_single_example(data, generate_features_description())
            merge_and_save(renderers=renderers, data=data, output_path=config["output_path"])


if __name__ == "__main__":
    main()
