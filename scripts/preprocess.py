
import os
import pickle
import sys
import yaml
from tqdm import tqdm

from absl import flags
from ml_collections import ConfigDict, config_flags

from sbi_stream import datasets


def main(config: ConfigDict):

    output_dir = os.path.join(config.root_out, config.name_out)
    os.makedirs(output_dir, exist_ok=True)

    # convert config to yaml and write to output dir
    config_dict = config.to_dict()
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    input_dir = os.path.join(config.root, config.name)

    print(f"Processing raw data from {input_dir} and saving to {output_dir}")
    for i in tqdm(range(config.start_dataset, config.start_dataset + config.num_datasets)):
        data = datasets.read_and_process_raw_datasets(
            input_dir,
            data_format=config.data_format,
            features=config.features,
            labels=config.labels,
            num_datasets=1,
            start_dataset=i,
            num_subsamples=config.get("num_subsamples", 1),
            **config.get("dataset_args", {})
        )

        if data is not None:
            data_out_path = os.path.join(output_dir, f'data.{i}.pkl')

            if config.data_format == 'matched_filter':
                data_dict = {
                    'signal': data[0],
                    'bg_lsst': data[1],
                    'bg_roman': data[2],
                    'labels': data[3],
                }
                with open(data_out_path, "wb") as f:
                    pickle.dump(data_dict, f)
            else:
                print(f"Saving processed data to {data_out_path}")
                with open(data_out_path, "wb") as f:
                    pickle.dump(data, f)
        else:
            print(f"Error processing dataset {i}, skipping...")

    print("Preprocessing complete.")

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config",
        None,
        "File path to the preprocess config.",
        lock_config=True,
    )
    FLAGS(sys.argv)
    main(config=FLAGS.config)
