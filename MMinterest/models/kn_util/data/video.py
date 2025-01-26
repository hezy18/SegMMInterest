import numpy as np
import torch
import os.path as osp
import subprocess
import os
import glob
from ..basic import map_async
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kn_util.data.collate import collect_features_from_sample_list, merge_list_to_tensor
from PIL import Image
from functools import partial
import numpy as np
from functools import partial
import io
import warnings
from contextlib import redirect_stdout
from pathlib import Path


class HFImageModelWrapper(nn.Module):
    """enable huggingface image transformer to be used for video inference"""

    def __init__(self, model, extractor, batch_size=16, use_cuda=True) -> None:
        super().__init__()
        self.model = model
        self.extractor = extractor
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    @torch.no_grad()
    def forward(self, raw_data):

        images = [Image.open(path).convert("RGB") for path in raw_data]
        dataloader = DataLoader(images,
                                batch_size=self.batch_size,
                                collate_fn=partial(self.extractor, return_tensors="pt"),
                                num_workers=8,
                                prefetch_factor=6,
                                pin_memory=True)
        outputs_list = []
        for inputs in dataloader:
            if self.use_cuda:
                inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            outputs = self.post_forward(outputs)
            outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            outputs_list += [outputs]

        outputs = merge_list_to_tensor(collect_features_from_sample_list(outputs_list), mode="concat")
        return outputs

    def post_forward(self, outputs):
        return outputs


class VisionCLIPWrapper(HFImageModelWrapper):

    def __init__(self,
                 model,
                 extractor,
                 batch_size=16,
                 spatial_reduce=4,
                 temporal_reduce=2,
                 pooling="avg",
                 use_cuda=True) -> None:
        super().__init__(model, extractor, batch_size, use_cuda)
        self.spatial_reduce = spatial_reduce
        self.temporal_reduce = temporal_reduce
        self.pooling = pooling

    def post_forward(self, outputs):
        last_hidden_state = outputs["last_hidden_state"]
        patch_size = int(np.sqrt(last_hidden_state.shape[1]))
        last_hidden_state = rearrange(last_hidden_state[:, 1:, :], "t (h w) d -> d t h w", h=patch_size,
                                      w=patch_size)[None, :]
        pooling = F.avg_pool3d if self.pooling == "avg" else F.max_pool3d
        spatial_reduce = self.spatial_reduce
        temporal_reduce = np.minimum(last_hidden_state.shape[2], self.temporal_reduce)
        last_hidden_state = pooling(last_hidden_state,
                                    kernel_size=(temporal_reduce, spatial_reduce, spatial_reduce),
                                    stride=(temporal_reduce, spatial_reduce, spatial_reduce))
        last_hidden_state = rearrange(last_hidden_state[0], "d t h w -> t h w d")

        outputs["last_hidden_state"] = last_hidden_state
        return outputs


class FFMPEG:

    @classmethod
    def single_video_process(cls, video_path, output_video_path, frame_scale=None, fps=None, quiet=True):
        flags = ""
        if quiet:
            flags += " -hide_banner -loglevel error"

        vf_flags = []
        if frame_scale:
            vf_flags += [f"scale={frame_scale}"]
        if fps:
            vf_flags += [f"fps=fps={fps}"]

        if len(vf_flags) > 0:
            flags += " -vf " + ",".join(vf_flags)

        cmd = f"ffmpeg -i {video_path} {flags} {output_video_path}"
        if not quiet:
            print(cmd)

        subprocess.run(cmd, shell=True)

    @classmethod
    def single_video_to_image(cls,
                              video_path,
                              cur_image_dir,
                              max_frame=None,
                              frame_scale=None,
                              fps=None,
                              quiet=True,
                              overwrite=False):
        # if not overwrite and osp.exists(osp.join(cur_image_dir, ".finish")):
        #     return
        os.makedirs(cur_image_dir, exist_ok=True)
        subprocess.run(f"rm -rf {cur_image_dir}/*", shell=True)
        flag = ""
        if quiet:
            flag += "-hide_banner -loglevel error "
        # filter:v
        filter_v = []
        if frame_scale:
            if isinstance(frame_scale, int):
                frame_scale = f"{frame_scale}:{frame_scale}"
            assert isinstance(frame_scale, str)
            filter_v += [f"scale={frame_scale}"]
        if fps:
            filter_v += [f"fps=fps={fps}"]
        if len(filter_v) > 0:
            flag += f" -filter:v {','.join(filter_v)}"
        if max_frame:
            flag += f"-frames:v {max_frame} "

        cmd = f"ffmpeg -i {video_path} {flag} {osp.join(cur_image_dir, '%04d.png')}"
        if not quiet:
            print(cmd)

        subprocess.run(
            cmd,
            shell=True,
        )
        # subprocess.run(f"cd {cur_image_dir} && touch .finish", shell=True)

    @classmethod
    def multiple_video_to_image_async(cls, video_dir, image_root, **kwargs):
        video_paths = glob.glob(video_dir + "*")

        def func_single(inputs):
            return cls.single_video_to_image(**inputs)

        args = []
        for video_path in video_paths:
            video_id = osp.basename(video_path)[:-4]
            cur_image_dir = osp.join(image_root, video_id)
            args += [dict(video_path=video_path, cur_image_dir=cur_image_dir, **kwargs)]

        map_async(args, func_single, num_process=64)


import yt_dlp
class YTDLPDownloader:
    @classmethod
    def download(cls, youtube_id, video_path, video_format="worst[ext=mp4][height>=224]", quiet=True):
        # scale should be conditon like "<=224" or ">=224"
        ydl_opts = {
            'ignoreerrors': True,
            'format': video_format,
            'outtmpl': video_path,
            'quiet': quiet,
            'noprogress': quiet
        }

        url = f"https://www.youtube.com/watch?v={youtube_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download(url)

        return error_code == 0
    
    @classmethod
    def load_to_buffer(cls, youtube_id, video_format="worst[ext=mp4][height>=224]", quiet=True):
        # refer to https://github.com/yt-dlp/yt-dlp/issues/3298
        ydl_opts = {
            'ignoreerrors': True,
            'format': video_format,
            'outtmpl': "-",
            'logtostderr': True,
            'quiet': quiet,
            'noprogress': quiet
        }

        buffer = io.BytesIO()
        with redirect_stdout(buffer), yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([youtube_id])
        
        buffer.seek(0)

        # write out the buffer for demonstration purposes
        # Path(f"{youtube_id}.mp4").write_bytes(buffer.getvalue())
        return buffer, error_code == 0
