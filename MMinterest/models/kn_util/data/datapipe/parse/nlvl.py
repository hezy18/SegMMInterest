from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torch.utils.data import functional_datapipe
import numpy as np
import os.path as osp
from operator import itemgetter


class TACoSParser(IterDataPipe):

    def __init__(self, src_pipeline) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline

    def __iter__(self):
        for json_data in self.src_pipeline:
            for video_id, annot in json_data.items():
                video_id = video_id[:-4]
                for idx, (sentence, timestamp) in enumerate(zip(annot["sentences"], annot["timestamps"])):
                    gt = np.array(timestamp) / annot["num_frames"]
                    text_id = f"{video_id}_{idx}"

                    yield dict(video_id=video_id, text_id=text_id, gt=gt, text=sentence)


class ActivityNetParser(IterDataPipe):

    def __init__(self, src_pipeline) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline

    def __iter__(self):
        for json_data in self.src_pipeline:
            for video_id, annot in json_data.items():
                duration = annot["duration"]
                for idx, (sentence, timestamp) in enumerate(zip(annot["sentences"], annot["timestamps"])):
                    gt = np.array(timestamp) / duration
                    text_id = f"{video_id}_{idx}"

                    yield dict(video_id=video_id, text_id=text_id, gt=gt, text=sentence, duration=duration)


def build_nlvl_parser(annot_dir, dataset, split="train"):
    annot_file = osp.join(annot_dir, split + ".json")
    annot_file_wrapper = IterableWrapper([annot_file])
    dataset_dp = annot_file_wrapper.open_files("r", encoding="utf-8").parse_json_files().map(itemgetter(1))
    if dataset == "tacos":
        dataset_dp = TACoSParser(dataset_dp)
    if dataset == "activitynet":
        dataset_dp = ActivityNetParser(dataset_dp)
    if dataset == "charades":
        if split == "val":
            split = "test"
        dataset_dp = ActivityNetParser(dataset_dp)  # processed to same format as activitynet

    return dataset_dp
