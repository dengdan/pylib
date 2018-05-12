from __future__ import absolute_import

try:
    import cv2
except:
    from . import log
    log.info('cv2 is unavailable, util.img can not be used.')


class VideoWriter(object):
    """
    An easy to use video writer.
    Example:
    with util.video.VideoWriter(path = 'lane.mp4') as writer:
    for frame in images:
        writer.add_frame(frame)

    """
    def __init__(self,  path, shape = None, fps = 20):
        """
        If the shape is None, the shape of the first frame to be written will be used as the shape for all frames.
        """
        import util
        self.path = util.io.get_absolute_path(path)
        self.fps = fps
        self.shape = shape
        self.writer = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_code, traceback):
        self.close()
        
    def __init_writer__(self, frame):
        if self.shape is None:
            self.shape = frame.shape
            
        self.size = tuple(self.shape[:2][::-1])
        self.writer = create_video_writer(self.path, self.shape, self.fps)
        
    def add_frame(self, frame):
        if self.writer is None:
            self.__init_writer__(frame)
            
        h, w = frame.shape[:2]
        if h != self.shape[0] or w != self.shape[1]:
            from . import img
            frame = img.resize(frame, self.size)
        add_frame(self.writer, frame)
        
    def close(self):
        if self.writer:
            close_video_writer(self.writer)
    
    def release(self):
        self.close()
        
def create_video_writer(path, shape, fps = 20):
    from . import io_
    path = io_.get_absolute_path(path)
    io_.make_parent_dir(path)
    h, w = shape[:2]
    import cv2
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    except:
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        h, w = shape[:2]
    return cv2.VideoWriter(path, fourcc, fps, (w,h))
    
def add_frame(writer, frame):
    writer.write(frame)
    
def close_video_writer(writer):
    writer.release()

def get_fps(video):
    if type(video) == str:
        path = util.io.get_absolute_path('~/temp/no-use/lane.mp4')
        video = cv2.VideoCapture(path)  
    
    try:
        fps = video.get(cv2.CAP_PROP_FPS)  
    except:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    return fps

class VideoReader():
    """
    Example:
    with VideoReader(path) as video:
        for frame in video:
            cv2.imshow('', frame)
            ....
    """
    def __init__(self, path):
        import util
        self.path = util.io.get_absolute_path(path)
        self._video = cv2.VideoCapture(self.path)
        self._fps = get_fps(self._video)
    
    @property
    def fps(self):
        return self._fps
    
    def __iter__(self):
        return self
    
    def __next__(self):
        success, frame = self._video.read()  
        if success:
            return frame
        raise StopIteration
    
    def next(self):
        return self.__next__()

    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_code, traceback):
        self._video.release()



        
        
        