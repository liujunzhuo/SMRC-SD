# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare the placeholder text dataset required by the agent trainer."""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="visual", choices=["visual", "text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--train_data_size", default=256, type=int)
    parser.add_argument("--val_data_size", default=256, type=int)
    args = parser.parse_args()

    print(f"processing data for mode: {args.mode}")
    local_dir = os.path.join(args.local_dir, args.mode)
    # This public Geometry3k dataset supplies only the required row count and
    # modality marker. ALFWorld/WebShop tasks are created by the environment.
    dataset = datasets.load_dataset("hiyouga/geometry3k")
    train_dataset = dataset["train"].select(range(args.train_data_size))
    test_dataset = dataset["test"].select(range(args.val_data_size))
    prompt = "<image>" if args.mode == "visual" else ""

    def make_map_fn(split):
        def process_fn(example, idx):
            images = example.pop("images")
            row = {
                "data_source": args.mode,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "agent",
                "extra_info": {"split": split, "index": idx},
            }
            if args.mode == "visual":
                row["images"] = images
            return row

        return process_fn

    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(make_map_fn("test"), with_indices=True, num_proc=8)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
