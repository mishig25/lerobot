#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
from io import StringIO
from pathlib import Path
import pandas as pd
import requests
import tempfile
import os
from safetensors import safe_open

import numpy as np
from flask import Flask, redirect, render_template, url_for

from lerobot.common.utils.utils import init_logging
import lerobot


def get_ep_csv_fname(episode_id: int):
    ep_csv_fname = f"episode_{episode_id}.csv"
    return ep_csv_fname


def read_remote_safetensors(url, framework="np", device="cpu"):
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name

    try:
        # Open the safetensors file
        with safe_open(temp_file_path, framework=framework, device=device) as f:
            # Read and return the keys
            keys = list(f.keys())
            tensors = {key: f.get_tensor(key) for key in keys}
            return tensors
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def get_video_columns(df):
    def is_dict_with_path(val):
        return isinstance(val, dict) and "path" in val

    # Apply the check to each element in the DataFrame
    mask = df.applymap(is_dict_with_path)

    # Find columns where all values satisfy the condition
    columns_with_path = mask.all().index[mask.all()].tolist()

    return columns_with_path


def get_episode_data_csv_str(df, data_index, episode_index):
    """Get a csv str containg timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    from_idx = data_index["from"][episode_index]
    to_idx = data_index["to"][episode_index]

    df_columns = df.columns.tolist()
    has_state = "observation.state" in df_columns
    has_action = "action" in df_columns

    # init header of csv with state and action names
    header = ["timestamp"]
    if has_state:
        dim_state = len(df["observation.state"][0])
        header += [f"state_{i}" for i in range(dim_state)]
    if has_action:
        dim_action = len(df["action"][0])
        header += [f"action_{i}" for i in range(dim_action)]

    columns = ["timestamp"]
    if has_state:
        columns += ["observation.state"]
    if has_action:
        columns += ["action"]

    data = df.loc[from_idx:to_idx, columns]
    rows = np.hstack(
        (np.expand_dims(data["timestamp"], axis=1), *[np.vstack(data[col]) for col in columns[1:]])
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string


def get_episode_video_paths(df: pd.DataFrame, data_index, repo_id: str, episode_index: int) -> list[str]:
    from_idx = data_index["from"][episode_index]
    video_columns = get_video_columns(df)
    paths = [val["path"] for val in df.loc[from_idx, video_columns].tolist()]
    paths = [f"https://huggingface.co/datasets/{repo_id}/resolve/main/{p}" for p in paths]
    return paths


def visualize_dataset_html(
    host: str = "127.0.0.1",
    port: int = 9090,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    app = Flask(__name__, template_folder=template_dir.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    @app.route("/")
    def hommepage():
        featured_datasets = [
            "cadene/koch_bimanual_folding",
            "lerobot/aloha_static_cups_open",
            "lerobot/columbia_cairlab_pusht_real",
        ]
        return render_template(
            "visualize_dataset_homepage.html",
            featured_datasets=featured_datasets,
            lerobot_datasets=lerobot.available_datasets,
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>")
    def show_first_episode(dataset_namespace, dataset_name):
        first_episode_id = 0
        return redirect(
            url_for(
                "show_episode",
                dataset_namespace=dataset_namespace,
                dataset_name=dataset_name,
                episode_id=first_episode_id,
            )
        )

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        url_data = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/train-00000-of-00001.parquet"
        url_index = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta_data/episode_data_index.safetensors"
        )

        data_index = read_remote_safetensors(url_index)
        df = pd.read_parquet(url_data)

        episode_data_csv_str = get_episode_data_csv_str(df, data_index, episode_id)

        dataset_info = {
            "repo_id": repo_id,
            "num_episodes": len(data_index["from"]),
        }
        video_paths = get_episode_video_paths(df, data_index, repo_id, episode_id)
        videos_info = [{"url": video_path, "filename": Path(video_path).name} for video_path in video_paths]

        episodes = [idx for idx in range(len(data_index["from"]))]

        return render_template(
            "visualize_dataset_template.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            has_policy=False,
            episode_data_csv_str=episode_data_csv_str,
        )

    app.run(host=host, port=port, debug=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )

    args = parser.parse_args()
    visualize_dataset_html(**vars(args))


if __name__ == "__main__":
    main()
