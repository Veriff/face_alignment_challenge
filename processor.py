import numpy as np

from base import ProcessorBase, ProcessorResult, load_data_by_id


class Processor(ProcessorBase):
    """ Use this class as a base for your solution.

    You'll want to re-implement most, if not all of the methods here.
    See BaselineProcessor class for example implementation which you're welcome to use.
    """

    def __init__(self, videos_csv_filename: str) -> None:
        super().__init__(videos_csv_filename)

        # You can do additional setup here, e.g. loading models, etc.

    def build_index(self, filename: str = None):
        """ Builds index for given input files

        This is run once per inputs and gives you a chance to do heavier pre-processing on the input data.
        We recommend using the filename as prefix for any index files that you create.
        """

        pass

    def load_index(self, filename: str):
        """ Load index that was built by build_index() for given input files
        """
        pass

    def reset(self) -> None:
        """ Called whenever we start processing a new video, unrelated to the previous frames

        You can use it to reset any temporary state such as last frame, statistics, etc.
        """

        pass

    def process_frame(self, frame: np.ndarray, landmarks: np.ndarray) -> ProcessorResult:
        """ The main method - takes an input frame and landmarks and returns the result

        Frame is a color image - a frame from a video. Has shape (height, width, channels) where channels = 3.
        Landmarks is numpy array of shape (68, 2), containing the detected face landmarks.

        Note that the frame in the output MUST have the same shape as the input frame.
        """

        raise NotImplementedError()
