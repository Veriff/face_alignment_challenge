import glob
import logging
from glob import glob
from pathlib import Path
from typing import Tuple

import attr
import cv2
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm

from videocaptureasync import VideoCaptureAsync

logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class ProcessorResult:
    # The resulting color image (one frame of video)
    frame: np.ndarray
    # Index of the frame in the original data
    frame_idx: int

    # Face landmarks which could be transformed the same way as `frame`
    landmarks: np.array


def overlay_landmarks_on_frame(landmarks, frame):
    """
    :param landmarks: 2d landmarks for the face
    :param frame: corresponding image
    :return: frame with face drawn on top
    """
    frame = frame.astype(np.uint8)
    landmarks = landmarks.astype(np.int32)
    # define which points need to be connected with a line
    jaw_points = [0, 17]
    right_eyebrow_points = [17, 22]
    left_eyebrow_points = [22, 27]
    nose_ridge_points = [27, 31]
    nose_base_points = [31, 36]
    right_eye_points = [36, 42]
    left_eye_points = [42, 48]
    outer_mouth_points = [48, 60]
    inner_mouth_points = [60, 68]

    connected_points = [
        right_eyebrow_points,
        left_eyebrow_points,
        nose_ridge_points,
        nose_base_points,
        right_eye_points,
        left_eye_points,
        outer_mouth_points,
        inner_mouth_points,
    ]

    unconnected_points = [jaw_points]

    for conPts in connected_points:
        frame = cv2.polylines(
            frame, [landmarks[conPts[0] : conPts[1]]], isClosed=True, color=[255, 255, 255], thickness=1
        )

    for conPts in unconnected_points:
        frame = cv2.polylines(
            frame, [landmarks[conPts[0] : conPts[1]]], isClosed=False, color=[255, 255, 255], thickness=1
        )

    return frame


def load_data(npz_filepath):
    """
    :param npz_filepath: .npz file from youtube-faces-with-keypoints dataset
    :return: color_images, bounding_box, landmarks_2d, landmarks_3d
    """
    with np.load(npz_filepath) as face_landmark_data:
        color_images = face_landmark_data["colorImages"]
        bounding_box = face_landmark_data["boundingBox"]
        landmarks_2d = face_landmark_data["landmarks2D"]
        landmarks_3d = face_landmark_data["landmarks3D"]

    return color_images, bounding_box, landmarks_2d, landmarks_3d


def full_cost_function(query_video, predicted_video, query_keypoints, predicted_keypoints):
    pass


def frame_cost_function(last_frame, current_frame, keypoints_query, keypoints_current, query_image):
    """
    :param last_frame: np.array of the last frame in predicted sequence
    :param current_frame: np.array of the current frame in predicted sequence
    :param keypoints_query: np.array of 2D keypoints on the current frame of query sequence
    :param keypoints_current: np.array of 2D keypoints on the current frame of predicted sequence

    :return: float, a weighted sum of different costs
    """
    image_diff_w = 0.1
    keypoint_diff_w = 0.9

    img_diff = np.mean(np.abs(cv2.resize(last_frame, current_frame.shape[:2][::-1]) - current_frame) > 25)

    # normalizing keypoints to be [0; 1]
    qshape = np.array(query_image.shape[:2])
    cshape = np.array(current_frame.shape[:2])

    keypoints_query_norm = keypoints_query / qshape
    keypoints_current_norm = keypoints_current / cshape

    keypoint_diff = np.mean(np.abs(keypoints_query_norm - keypoints_current_norm)) * 10

    return float(img_diff) * image_diff_w, float(keypoint_diff) * keypoint_diff_w


class FaceEmbeddingGenerator:
    """
    Base class of embedding generators
    """

    dim: int

    @staticmethod
    def make_embedding(embedding):
        return embedding.flatten()


class FaceEmbeddingGenerator2D(FaceEmbeddingGenerator):
    """ Embedding generator for 3D keypoints, which holds statically the input dimensions"""

    dim = 68 * 2


class FaceEmbeddingGenerator3D(FaceEmbeddingGenerator):
    """ Embedding generator for 2D keypoints, which holds statically the input dimensions"""

    dim = 68 * 3


def get_embedding_maker() -> FaceEmbeddingGenerator:
    return FaceEmbeddingGenerator2D()


def get_annoy_index(embedding_maker: FaceEmbeddingGenerator) -> AnnoyIndex:
    landmarks_index_args = [embedding_maker.dim, "euclidean"]
    return AnnoyIndex(*landmarks_index_args)  # Approximate search index


def load_data_by_id(id: int, video_df):
    """Given an frame ID, and a dataset description"""
    video = video_df[(video_df.start < id) & (video_df.end > id)]
    if len(video) == 0:
        return None, None, None, None

    paths = glob("./data/*/{videoID}.npz".format(videoID=video.iloc[0].videoID))
    if len(paths) == 0:
        return None, None, None, None

    path = paths[0]
    frame_id = int(id - video.start)
    q_color_images, q_bounding_box, q_landmarks_2d, q_landmarks_3d = load_data(path)
    return (
        q_color_images[..., frame_id],
        q_bounding_box[..., frame_id],
        q_landmarks_2d[..., frame_id],
        q_landmarks_3d[..., frame_id],
    )


def load_image_by_id(id, video_df):
    """Utility function to get a single frame by ID"""
    return load_data_by_id(id, video_df)[0]


class OutputWriter:
    def __init__(self, filename: str, width: int, height: int) -> None:
        super().__init__()

        self.fps = 25
        self.filename = filename
        self.frame_no = 0

        fourcc = cv2.VideoWriter_fourcc(*"X264")
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        if not self.video_writer.isOpened():
            logger.warning("Couldn't create OpenCV video writer, you're probably missing dependencies (see README)")
            logger.warning("Will write individual frames to %s-frames/ instead", filename)
            self.video_writer = None

    def write_frame(self, frame: np.ndarray):
        self.frame_no += 1
        if self.video_writer is not None:
            self.video_writer.write(frame)
        else:
            frames_dir = Path(f"{self.filename}-frames")
            frames_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames_dir / f"{self.frame_no:04d}.jpg"), frame)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()


class ProcessorBase:
    default_index_filename = "data/index"

    def __init__(self, videos_csv_filename: str) -> None:
        self.embedding_maker = get_embedding_maker()
        self.video_df = pd.read_csv(videos_csv_filename)

    def build_index(self, filename: str):
        pass

    def load_index(self, filename: str):
        pass

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks: np.ndarray) -> ProcessorResult:
        raise NotImplementedError()

    def process_video(self, video_filename: str, output_filename: str = "output.avi"):
        self.reset()

        q_color_images, q_bounding_box, q_landmarks_2d, q_landmarks_3d = load_data(video_filename)
        logger.info("color shape: %s", q_color_images.shape)

        last_predicted_image = q_color_images[..., 0]
        output_w, output_h = self.get_output_frame_size(q_color_images.shape)
        output_writer = OutputWriter(output_filename, output_w, output_h)
        try:
            for i in tqdm(range(q_color_images.shape[-1])):
                query_image = q_color_images[..., i]
                landmarks = q_landmarks_2d[..., i]
                result = self.process_frame(frame=query_image, landmarks=landmarks)

                if result is None or result.frame is None:
                    logger.error("Got missing or invalid result for frame %d", i)
                    continue
                if result.frame.shape != query_image.shape:
                    logger.error(
                        "Result frame has different shape %s than input frame %s, skipping",
                        result.frame.shape,
                        query_image.shape,
                    )
                    continue

                frame_costs = frame_cost_function(
                    last_predicted_image, result.frame, q_landmarks_2d[..., i], result.landmarks, query_image
                )
                last_predicted_image = result.frame

                output_frame = self.create_output_frame(query_image, landmarks, result, i, frame_costs)
                output_writer.write_frame(output_frame)

        finally:
            output_writer.close()

    def process_webcam(self, use_cuda: bool = False):
        logger.info("Loading face-alignment model")
        try:
            import face_alignment
        except ImportError:
            logger.error("Please install face-alignment package to use webcam input:")
            logger.error("    pip install face-alignment==1.0.0")
            return

        device = "cuda" if use_cuda else "cpu"
        logger.info("Using device %s", device)
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

        self.reset()

        logger.info("Initializing OpenCV video stream")
        stream = VideoCaptureAsync(0)
        stream.start()
        try:
            if stream.cap.isOpened():  # try to get the first frame
                _, frame = stream.read()
            else:
                logger.error("Couldn't open webcam stream")
                return

            opencv_window_name = "face-alignment"
            cv2.namedWindow(opencv_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(opencv_window_name, 1000, 500)

            last_predicted_image = frame
            while True:
                logger.info("Reading frame")
                ok, frame = stream.read()
                if not ok:
                    break

                logger.info("Got frame %s", frame.shape)

                preds = fa.get_landmarks(frame)
                if preds:
                    # TODO: select the largest face
                    landmarks = preds[0]

                    result = self.process_frame(frame=frame, landmarks=landmarks)

                    if result is None or result.frame is None:
                        logger.error("Got missing or invalid result")
                        continue
                    if result.frame.shape != frame.shape:
                        logger.error(
                            "Result frame has different shape %s than input frame %s, skipping",
                            result.frame.shape,
                            frame.shape,
                        )
                        continue

                    frame_costs = frame_cost_function(
                        last_predicted_image, result.frame, landmarks, result.landmarks, frame
                    )

                    output_frame = self.create_output_frame(frame, landmarks, result, 0, frame_costs)

                else:
                    output_frame = frame

                cv2.imshow(opencv_window_name, output_frame)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    logger.info("Quitting")
                    break
                elif key == ord("f"):
                    is_fs = cv2.getWindowProperty(opencv_window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(
                        opencv_window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if not is_fs else cv2.WINDOW_NORMAL,
                    )

        finally:
            stream.stop()

    def get_output_frame_size(self, input_images_shape: tuple) -> Tuple[int, int]:
        """ Returns size (width, height) of the output frames

        Override this and create_output_frame() to make custom visualizations.
        """

        image_w = input_images_shape[1]
        image_h = input_images_shape[0]
        return image_w * 2, image_h

    def create_output_frame(
        self, query_image: np.ndarray, landmarks: np.ndarray, result: ProcessorResult, query_index: int, costs
    ) -> np.ndarray:
        input_with_landmarks = overlay_landmarks_on_frame(landmarks, query_image)
        output_with_landmarks = overlay_landmarks_on_frame(result.landmarks, result.frame)

        viz = np.concatenate((input_with_landmarks, output_with_landmarks), axis=1)
        text = f"Query: {query_index}; target: {result.frame_idx}; cost: {costs[0]:.3f} / {costs[1]:.3f}"
        cv2.putText(viz, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return viz
