# import av
# import math
# import numpy as np
# import gymnasium as gym
# from typing import List, Tuple, Optional, Union


# def get_accumulate_timestamp_idxs(
#     timestamps: List[float],  
#     start_time: float, 
#     dt: float, 
#     eps:float=1e-5,
#     next_global_idx: Optional[int]=0,
#     allow_negative=False
#     ) -> Tuple[List[int], List[int], int]:
#     """
#     For each dt window, choose the first timestamp in the window.
#     Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
#     next_global_idx should start at 0 normally, and then use the returned next_global_idx. 
#     However, when overwiting previous values are desired, set last_global_idx to None.

#     Returns:
#     local_idxs: which index in the given timestamps array to chose from
#     global_idxs: the global index of each chosen timestamp
#     next_global_idx: used for next call.
#     """
#     local_idxs = list()
#     global_idxs = list()
#     for local_idx, ts in enumerate(timestamps):
#         # add eps * dt to timestamps so that when ts == start_time + k * dt 
#         # is always recorded as kth element (avoiding floating point errors)
#         global_idx = math.floor((ts - start_time) / dt + eps)
#         if (not allow_negative) and (global_idx < 0):
#             continue
#         if next_global_idx is None:
#             next_global_idx = global_idx

#         n_repeats = max(0, global_idx - next_global_idx + 1)
#         for i in range(n_repeats):
#             local_idxs.append(local_idx)
#             global_idxs.append(next_global_idx + i)
#         next_global_idx += n_repeats
#     return local_idxs, global_idxs, next_global_idx


# class VideoRecorder:
#     def __init__(self,
#         fps,
#         codec,
#         input_pix_fmt,
#         # options for codec
#         **kwargs
#     ):
#         """
#         input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
#         """

#         self.fps = fps
#         self.codec = codec
#         self.input_pix_fmt = input_pix_fmt
#         self.kwargs = kwargs
#         # runtime set
#         self._reset_state()
    
#     def _reset_state(self):
#         self.container = None
#         self.stream = None
#         self.shape = None
#         self.dtype = None
#         self.start_time = None
#         self.next_global_idx = 0
    
#     @classmethod
#     def create_h264(cls,
#             fps,
#             codec='h264',
#             input_pix_fmt='rgb24',
#             output_pix_fmt='yuv420p',
#             crf=18,
#             profile='high',
#             **kwargs
#         ):
#         obj = cls(
#             fps=fps,
#             codec=codec,
#             input_pix_fmt=input_pix_fmt,
#             pix_fmt=output_pix_fmt,
#             options={
#                 'crf': str(crf),
#                 'profile': profile
#             },
#             **kwargs
#         )
#         return obj


#     def __del__(self):
#         self.stop()

#     def is_ready(self):
#         return self.stream is not None

#     def start(self, file_path, start_time=None):
#         if self.is_ready():
#             # if still recording, stop first and start anew.
#             self.stop()

#         self.container = av.open(file_path, mode='w')
#         self.stream = self.container.add_stream(self.codec, rate=self.fps)
#         codec_context = self.stream.codec_context
#         for k, v in self.kwargs.items():
#             setattr(codec_context, k, v)
#         self.start_time = start_time
    
#     def write_frame(self, img: np.ndarray, frame_time=None):
#         if not self.is_ready():
#             raise RuntimeError('Must run start() before writing!')
        
#         n_repeats = 1
#         if self.start_time is not None:
#             local_idxs, global_idxs, self.next_global_idx \
#                 = get_accumulate_timestamp_idxs(
#                 # only one timestamp
#                 timestamps=[frame_time],
#                 start_time=self.start_time,
#                 dt=1/self.fps,
#                 next_global_idx=self.next_global_idx
#             )
#             # number of appearance means repeats
#             n_repeats = len(local_idxs)
        
#         if self.shape is None:
#             self.shape = img.shape
#             self.dtype = img.dtype
#             h,w,c = img.shape
#             self.stream.width = w
#             self.stream.height = h
#         assert img.shape == self.shape
#         assert img.dtype == self.dtype

#         frame = av.VideoFrame.from_ndarray(
#             img, format=self.input_pix_fmt)
#         for i in range(n_repeats):
#             for packet in self.stream.encode(frame):
#                 self.container.mux(packet)

#     def stop(self):
#         if not self.is_ready():
#             return

#         # Flush stream
#         for packet in self.stream.encode():
#             self.container.mux(packet)

#         # Close the file
#         self.container.close()

#         # reset runtime parameters
#         self._reset_state()


# class VideoRecordingWrapper(gym.Wrapper):
#     def __init__(self, 
#             env, 
#             video_recoder: VideoRecorder,
#             mode='rgb_array',
#             file_path=None,
#             steps_per_render=1,
#             **kwargs
#         ):
#         """
#         When file_path is None, don't record.
#         """
#         super().__init__(env)
        
#         self.mode = mode
#         self.render_kwargs = kwargs
#         self.steps_per_render = steps_per_render
#         self.file_path = file_path
#         self.video_recoder = video_recoder

#         self.step_count = 0

#     def reset(self, **kwargs):
#         obs = super().reset(**kwargs)
#         self.frames = list()
#         self.step_count = 1
#         self.video_recoder.stop()
#         return obs
    
#     def step(self, action):
#         result = super().step(action)
#         self.step_count += 1
#         if self.file_path is not None \
#             and ((self.step_count % self.steps_per_render) == 0):
#             if not self.video_recoder.is_ready():
#                 self.video_recoder.start(self.file_path)

#             frame = self.env.render(**self.render_kwargs)
#             assert frame.dtype == np.uint8
#             self.video_recoder.write_frame(frame)
#         return result
    
#     def render(self, mode='rgb_array', **kwargs):
#         if self.video_recoder.is_ready():
#             self.video_recoder.stop()
#         return self.file_path
    

import imageio
import numpy as np
import gymnasium as gym
from typing import Tuple, Union
from icon.utils.file_utils import str2path, mkdir


class VideoRecordingWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        video_save_dir: Union[str, None] = None
    ) -> None:
        super().__init__(env)
        self.global_step = 0
        if video_save_dir is None:
            self.record_videos = False
        else:
            self.record_videos = True
            self.video_save_dir = mkdir(str2path(video_save_dir), parents=True, exist_ok=True)
            self.frames = list()
            
    def reset(self, **kwargs) -> None:
        obs = super().reset(**kwargs)
        self.global_step += 1
        if self.record_videos:
            self.frames = list()
            self.video_recoder = imageio.get_writer(self.video_save_dir.joinpath(f"trial_{str(self.global_step).zfill(3)}.mp4"), fps=24)
        return obs
    
    def step(self, action: np.ndarray) -> Tuple:
        outputs = super().step(action)
        if self.record_videos:
            frame = self.env.render()
            assert frame.dtype == np.uint8
            self.video_recoder.append_data(frame)
        return outputs
    
    def close(self) -> None:
        super().close()