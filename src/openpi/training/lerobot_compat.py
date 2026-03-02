import os
from pprint import pformat

import numpy as np
import torch


def _to_tensor_list(values) -> list[torch.Tensor]:
    if hasattr(values, "to_pylist"):
        values = values.to_pylist()
    elif isinstance(values, torch.Tensor):
        values = [values]
    else:
        values = list(values)

    return [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in values]


def apply_lerobot_compatibility(*, skip_video_decode: bool = False) -> None:
    """Patch LeRobot runtime behavior for current environment compatibility.

    - `datasets` may return `Column` instead of Python lists.
    - Some environments lack a working pyav backend for torchvision VideoReader.
    """
    from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset_mod
    from lerobot.common.datasets import utils as _lerobot_utils_mod
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not getattr(LeRobotDataset, "_openpi_column_patch", False):
        def _column_tensor(hf_dataset, key: str, *, dtype: torch.dtype) -> torch.Tensor:
            # Use the underlying Arrow table to avoid triggering `set_transform`,
            # which can be extremely slow for full-column access.
            table = hf_dataset.data.table
            values = table[key].to_pylist()
            return torch.as_tensor(np.asarray(values), dtype=dtype)

        def _check_timestamps_sync(
            hf_dataset,
            episode_data_index: dict[str, torch.Tensor],
            fps: int,
            tolerance_s: float,
            raise_value_error: bool = True,
        ) -> bool:
            timestamps = _column_tensor(hf_dataset, "timestamp", dtype=torch.float64)
            diffs = torch.diff(timestamps)
            within_tolerance = torch.abs(diffs - 1 / fps) <= tolerance_s

            mask = torch.ones(len(diffs), dtype=torch.bool)
            ignored_diffs = episode_data_index["to"][:-1] - 1
            mask[ignored_diffs] = False
            filtered_within_tolerance = within_tolerance[mask]

            if not torch.all(filtered_within_tolerance):
                original_indices = torch.arange(len(diffs))
                filtered_indices = original_indices[mask]
                outside_tolerance_filtered_indices = torch.nonzero(~filtered_within_tolerance)
                outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]
                episode_indices = _column_tensor(hf_dataset, "episode_index", dtype=torch.long)

                outside_tolerances = []
                for idx in outside_tolerance_indices:
                    outside_tolerances.append(
                        {
                            "timestamps": [timestamps[idx], timestamps[idx + 1]],
                            "diff": diffs[idx],
                            "episode_index": episode_indices[idx].item(),
                        }
                    )

                if raise_value_error:
                    raise ValueError(
                        "One or several timestamps unexpectedly violate the tolerance inside episode range.\n"
                        f"{pformat(outside_tolerances)}"
                    )
                return False

            return True

        # `check_timestamps_sync` is imported into lerobot_dataset at module import time,
        # so patch both module symbols.
        # _lerobot_utils_mod.check_timestamps_sync = _check_timestamps_sync
        # _lerobot_dataset_mod.check_timestamps_sync = _check_timestamps_sync

        def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
            result = {}
            for key, q_idx in query_indices.items():
                if key in self.meta.video_keys:
                    continue
                result[key] = torch.stack(_to_tensor_list(self.hf_dataset.select(q_idx)[key]))
            return result

        def _get_query_timestamps(self, current_ts: float, query_indices: dict[str, list[int]] | None = None) -> dict:
            query_timestamps = {}
            for key in self.meta.video_keys:
                if query_indices is not None and key in query_indices:
                    timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                    query_timestamps[key] = torch.stack(_to_tensor_list(timestamps)).tolist()
                else:
                    query_timestamps[key] = [current_ts]
            return query_timestamps

        LeRobotDataset._query_hf_dataset = _query_hf_dataset
        LeRobotDataset._get_query_timestamps = _get_query_timestamps
        LeRobotDataset._openpi_column_patch = True

    if skip_video_decode and not getattr(LeRobotDataset, "_openpi_skip_video_decode_patch", False):
        def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
            del ep_idx  # Unused: frames are synthetic.
            item = {}
            for vid_key, query_ts in query_timestamps.items():
                h, w, c = self.meta.features[vid_key]["shape"]
                frames = torch.zeros((len(query_ts), c, h, w), dtype=torch.float32)
                item[vid_key] = frames.squeeze(0) if len(query_ts) == 1 else frames
            return item

        LeRobotDataset._query_videos = _query_videos
        LeRobotDataset._openpi_skip_video_decode_patch = True


def maybe_apply_from_env() -> None:
    if os.getenv("OPENPI_LEROBOT_COLUMN_COMPAT", "0") != "1":
        return

    skip_video_decode = os.getenv("OPENPI_LEROBOT_SKIP_VIDEO_DECODE", "0") == "1"
    apply_lerobot_compatibility(skip_video_decode=skip_video_decode)
