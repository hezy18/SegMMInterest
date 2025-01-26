from ..video import FFMPEG, YTDLPDownloader
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import os
import os.path as osp
import glob
import numpy as np
from PIL import Image
import subprocess
from decord import VideoReader
import warnings


@functional_datapipe("download_youtube")
class YoutubeDownloader(IterDataPipe):

    def __init__(self,
                 src_pipeline,
                 cache_dir=None,
                 from_key=None,
                 to_file=False,
                 to_buffer=False,
                 quiet=True,
                 **downloader_args) -> None:
        # use pytube by default, ytdlp deprecated
        # load: return byteio
        # dump: return filepath
        assert to_file or to_buffer
        assert not to_file or cache_dir
        self.src_pipeline = src_pipeline
        self.to_file = to_file
        self.to_buffer = to_buffer
        self.cache_dir = cache_dir
        self.from_key = from_key
        self.downloader_args = downloader_args
        # self.downloader = downloader
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __iter__(self):
        for x in self.src_pipeline:
            youtube_id = x if not self.from_key else x[self.from_key]
            if self.to_buffer and not self.to_file:
                buffer, success = YTDLPDownloader.load_to_buffer(youtube_id, **self.downloader_args)
                if not success:
                    continue
                x.update({self.from_key + ".buffer": buffer})
                yield x
                buffer.close()
            else:
                ret_dict = dict()
                video_fn = osp.join(self.cache_dir, f"{youtube_id}.raw.mp4")
                # PyTubeDownloader.download(vid=youtube_id, fn=video_fn, **self.downloader_args)
                success = YTDLPDownloader.download(youtube_id=youtube_id, video_path=video_fn, quiet=True)
                if not success:
                    continue
                if self.to_file:
                    ret_dict[self.from_key + ".path"] = video_fn
                if self.to_buffer:
                    ret_dict[self.from_key + ".buffer"] = open(video_fn, "rb")

                x.update(ret_dict)
                yield x

                if self.to_buffer:
                    ret_dict[self.from_key + ".buffer"].close()


@functional_datapipe("ffmpeg_to_video")
class FFMPEGToVideo(IterDataPipe):

    def __init__(self, src_pipeline, from_key=None, remove_cache=True, **ffmpeg_args) -> None:
        self.src_pipeline = src_pipeline
        self.from_key = from_key
        self.remove_cache = remove_cache
        self.ffmpeg_args = ffmpeg_args

    def __iter__(self):
        for x in self.src_pipeline:
            video_path_raw = x[self.from_key]
            cache_dir = osp.dirname(video_path_raw)
            fn_no_extension, ext = osp.basename(video_path_raw).rsplit('.', maxsplit=1)
            video_path = osp.join(cache_dir, fn_no_extension + f".ffmpeg.{ext}")
            FFMPEG.single_video_process(video_path_raw, video_path, **self.ffmpeg_args)

            x[self.from_key + ".ffmpeg"] = video_path
            yield x

            if self.remove_cache:
                subprocess.Popen(f"rm -rf {video_path}", shell=True)


@functional_datapipe("load_frames_decord")
class DecordFrameLoader(IterDataPipe):

    def __init__(self,
                 src_pipeline,
                 stride=1,
                 from_key=None,
                 width=224,
                 height=224,
                 to_array=False,
                 to_images=False) -> None:
        # decord_args: width=224, height=224
        self.src_pipeline = src_pipeline
        self.from_key = from_key
        self.width = width
        self.height = height
        self.stride = stride
        self.to_images = to_images
        self.to_array = to_array


    def __iter__(self):
        for x in self.src_pipeline:
            buffer = x[self.from_key] if self.from_key else x
            vr = VideoReader(buffer, width=self.width, height=self.height)

            indices = list(range(0, len(vr), self.stride))
            arr = vr.get_batch(indices).asnumpy()

            if self.to_array:
                x[self.from_key + ".frame_arr"] = arr
            if self.to_images:
                x[self.from_key + ".frame_img"] = [Image.fromarray(_) for _ in arr]
            yield x


@functional_datapipe("load_frames_ffmpeg")
class FFMPEGFrameLoader(IterDataPipe):

    def __init__(self,
                 src_pipeline,
                 from_key=None,
                 remove_cache=True,
                 to_images=False,
                 to_array=False,
                 **ffmpeg_args) -> None:
        self.src_pipeline = src_pipeline
        self.from_key = from_key
        self.remove_cache = remove_cache
        self.ffmpeg_args = ffmpeg_args
        self.to_array = to_array
        self.to_images = to_images

    def __iter__(self):
        for x in self.src_pipeline:
            video_path_raw = x[self.from_key]
            cache_dir = osp.dirname(video_path_raw)
            fn_no_extension, ext = osp.basename(video_path_raw).rsplit('.', maxsplit=1)
            image_dir = osp.join(cache_dir, fn_no_extension)
            FFMPEG.single_video_to_image(video_path_raw, image_dir, **self.ffmpeg_args)

            image_paths = glob.glob(osp.join(image_dir, "*.png"))
            frames = [Image.open(fn).convert("RGB") for fn in image_paths]
            if self.to_array:
                x[self.from_key + ".frame_arr"] = np.stack([np.asarray(_) for _ in frames])
            if self.to_images:
                x[self.from_key + ".frame_img"] = frames

            yield x

            if self.remove_cache:
                subprocess.Popen(f"rm -rf {image_dir}", shell=True)