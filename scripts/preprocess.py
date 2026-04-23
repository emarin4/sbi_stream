
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
        data = datasets.read_raw_particle_datasets(
            input_dir, config.features, config.labels,
            num_datasets=1,
            init=i,
            num_subsamples=config.get("num_subsamples", 1),
            num_per_subsample=config.get("num_per_subsample", None),
            num_per_subsample_min=config.get("num_per_subsample_min", None),
            num_per_subsample_max=config.get("num_per_subsample_max", None),
            phi1_min=config.phi1_min,
            phi1_max=config.phi1_max,
            uncertainty_model=config.get('uncertainty_model', None),
            include_uncertainty=config.get('include_uncertainty', False),
        )
        if data is not None:
            data_out_path = os.path.join(output_dir, f'data.{i}.pkl')
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
