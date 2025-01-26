from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
import numpy as np

class HCSTVGParser(IterDataPipe):
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

class VidSTGParser(IterDataPipe):
    def __init__(self, src_ppl) -> None:
        super().__init__()
        self.src_ppl = src_ppl
    
    def __iter__(self):
        for json_data in self.src_ppl:
            for video_id, annot in json_data.items()
