"""
code to read synchronized RGB and depth images

Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
from videoio import Uint16Reader, uint16read, VideoReader, videoread
import imageio
import numpy as np
import json
import os.path as osp


class VideoController:
    """
    load color or depth frame based on synchronized timestamp
    """
    def __init__(self, video_path, pre_load=False, time_only=False):
        """

        :param video_path: *.[kinect id].[color|depth-reg].mp4
        :param pre_load: load all video data into memory or not, suitable if run in server
        """
        assert '.depth-reg.mp4' in video_path or '.color.mp4' in video_path, 'the given path is not valid: {}'.format(video_path)
        self.video_path = video_path
        if '.depth-reg.mp4' in video_path:
            self.video_type = 'depth'
            self.time_file = video_path.replace('.depth-reg.mp4', '.time.json')
        else:
            self.video_type = 'color'
            self.time_file = video_path.replace('.color.mp4', '.time.json')

        self.frame_times = np.array(json.load(open(self.time_file))[self.video_type], dtype=float)/1e6
        self.kinect_id = int(osp.basename(video_path).split('.')[1])
        assert self.kinect_id in [0, 1, 2, 3], 'invalid kinect id encounted: {}'.format(self.kinect_id)

        if not time_only:
            # load video as well
            self.pre_load = pre_load
            if self.video_type == 'depth':
                if self.pre_load:
                    self.image_data = uint16read(self.video_path)
                    self.reader = None
                else:
                    self.reader = Uint16Reader(self.video_path)
                    self.image_data = None
            else:
                if self.pre_load:
                    self.reader = None
                    self.image_data = videoread(self.video_path)
                else:
                    self.reader = VideoReader(self.video_path)
                    # self.reader = imageio.get_reader(self.video_path, 'ffmpeg')
                    self.image_data = None
            self.video_iter = iter(self.reader) if not self.pre_load else None
        else:
            self.reader = None
        # data buffer
        self.cached_frame = None
        self.current_frame = 0

    def start_time(self):
        return self.frame_times[0]

    def end_time(self):
        return self.frame_times[-1]

    def get_closest_time(self, time):
        """
        the closest actual recording time to the query time
        """
        if time < 0 or time > self.end_time():
            return None
        time_diff = np.abs(self.frame_times - time)
        index = np.argmin(time_diff)
        return self.frame_times[index]

    def get_closest_frameidx(self, time, frame_times):
        """
        return the frame index of the closest frame
        """
        if time < self.start_time()-0.3 or time > self.end_time():
            print(f'given timestamp invalid for start time: {self.start_time()}, end time: {self.end_time()}')
            return None
        time_diff = np.abs(frame_times - time)
        index = np.argmin(time_diff)
        return index

    def get_closest_frame(self, time):
        """
        find the closest frame saved in video given the query time
        :param time: query time
        :return: color or depth image that is closest to the query time
        """
        frame_idx = self.get_closest_frameidx(time, self.frame_times)
        if frame_idx is None:
            raise ValueError('the given timestamp is invalid: {}'.format(time))

        if self.pre_load:
            return self.image_data[frame_idx]
        
        # if self.video_type == 'color':
        #     return np.array(self.reader.get_data(frame_idx-1)) # imageio index should be offset by 1

        frame_delta = frame_idx - self.current_frame
        self.current_frame = frame_idx
        if frame_delta < 0:
            raise Exception("Attempted to read kinect video backward")
        if frame_delta == 0 and self.cached_frame is not None:
            return self.cached_frame
        else:
            for _ in range(frame_delta - 1):
                _ = next(self.video_iter)
            img = next(self.video_iter)
            img = np.array(img)
        self.cached_frame = img
        return img

    def close(self):
        if self.reader is not None:
            self.reader.close()


class ColorDepthController:
    def __init__(self, kinect_video_prefix, kinect_id, pre_load=False):
        self.depth_path = kinect_video_prefix + f'.{kinect_id}.depth-reg.mp4'
        self.color_path = kinect_video_prefix + f'.{kinect_id}.color.mp4'
        self.color_reader = VideoController(self.color_path, pre_load)
        self.depth_reader = VideoController(self.depth_path, pre_load)

    def get_closest_frame(self, time):
        """
        return color and depth image that is closest to the given frame time
        :param time:
        :return:
        """
        color = self.color_reader.get_closest_frame(time)
        depth = self.depth_reader.get_closest_frame(time)

        return color, depth

    def start_time(self):
        return max(self.depth_reader.start_time(), self.color_reader.start_time())

    def end_time(self):
        return min(self.depth_reader.end_time(), self.color_reader.end_time())

    def get_closest_time(self, time):
        return self.depth_reader.get_closest_time(time)

    def close(self):
        self.depth_reader.close()
        self.color_reader.close()
