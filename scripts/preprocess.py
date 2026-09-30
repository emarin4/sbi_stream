
import os
import pickle
import sys
import yaml
from tqdm import tqdm
import re
from pathlib import Path

from absl import flags
from ml_collections import ConfigDict, config_flags

from sbi_stream import datasets

# def _extract_index(path: Path) -> int:
#     """Extract the integer N from a filename like 'data_N.h5'."""
#     match = re.search(r"data_(\d+)\.h5$", path.name)
#     if match is None:
#         raise ValueError(f"Filename {path.name} does not match expected 'data_N.h5' pattern")
#     return int(match.group(1))

def _extract_index(path: Path) -> int:
    """Extract a unique sortable index from
    'perturbers_batch_NNNN_streams_XXXXX_YYYYY.h5'."""
    match = re.search(r"perturbers_batch_(\d+)_streams_(\d+)_(\d+)\.h5$", path.name)
    if match is None:
        raise ValueError(f"Filename {path.name} does not match expected pattern")
    batch, start, end = match.groups()
    chunk_num = int(start) // 1000  # 00000->0, 01000->1, ..., 09000->9
    return int(batch) * 10 + chunk_num

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

    all_h5_files = sorted(Path(input_dir).glob("perturbers_batch_*_streams_*.h5"), key=_extract_index)
    print(f"Found {len(all_h5_files)} h5 files in {input_dir}")
    if len(all_h5_files) == 0:
        raise RuntimeError(f"No .h5 files found in {input_dir} — check glob pattern / directory path.")

    for p in all_h5_files:
        print(f"  index {_extract_index(p)} -> {p.name}")
    
    for i in tqdm(range(config.start_dataset, config.start_dataset + config.num_datasets)):
        if i >= len(all_h5_files):
            print(f"Index {i} out of range ({len(all_h5_files)} files found), skipping...")
            continue
            
        target_file = all_h5_files[i].name
        data = datasets.read_raw_particle_datasets_h5(
            input_dir, config.features, config.labels,
            h5_files=[target_file],
            #num_datasets=1,
            #init=i,
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
