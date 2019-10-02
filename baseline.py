import glob
import logging
from glob import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from base import ProcessorBase, ProcessorResult, get_annoy_index, load_data, load_data_by_id

logger = logging.getLogger(__name__)


class BaselineProcessor(ProcessorBase):
    def __init__(self, videos_csv_filename: str) -> None:
        super().__init__(videos_csv_filename)

        self.landmarks_index = get_annoy_index(self.embedding_maker)

    def build_index(self, filename: str):
        face_counter = 0
        logger.info("Loading videos...")
        for video_i, row in tqdm(self.video_df.iterrows(), total=len(self.video_df)):
            db_paths = glob("./data/*/{videoID}.npz".format(videoID=self.video_df.loc[video_i].videoID))
            if len(db_paths) == 0:
                continue

            db_path = db_paths[0]
            db_color_images, db_bounding_box, db_landmarks_2d, db_landmarks_3d = load_data(db_path)

            start_index = face_counter
            for frame_i in range(db_color_images.shape[-1]):
                face_counter += 1
                self.landmarks_index.add_item(
                    face_counter, self.embedding_maker.make_embedding(db_landmarks_2d[..., frame_i])
                )
            end_index = face_counter

            self.video_df.at[video_i, "start"] = start_index
            self.video_df.at[video_i, "end"] = end_index

        logger.info("Building index...")
        self.landmarks_index.build(10)  # 10 trees

        # Save the landmarks index
        landmarks_filename = f"{filename}.landmarks"
        logger.info("Saving landmarks index to %s", landmarks_filename)
        self.landmarks_index.save(landmarks_filename)

        # Save the updated CSV containing start and end for each video
        csv_filename = f"{filename}.landmarks.ann"
        logger.info("Saving CSV to %s", csv_filename)
        self.video_df.to_csv(csv_filename, index=False)

    def load_index(self, filename: str):
        csv_filename = f"{filename}.landmarks.ann"
        self.video_df = pd.read_csv(csv_filename)
        landmarks_filename = f"{filename}.landmarks"
        self.landmarks_index.load(landmarks_filename)  # super fast, will just mmap the file

    def reset(self) -> None:
        pass

    def process_frame(self, frame: np.ndarray, landmarks) -> ProcessorResult:
        nns, dists = self.landmarks_index.get_nns_by_vector(
            self.embedding_maker.make_embedding(landmarks), 10, include_distances=True
        )

        best_matches = [(image_i, dist) for image_i, dist in zip(nns, dists)]
        image_diffs = sorted(best_matches, key=lambda x: x[1], reverse=True)  # sort by distance

        best_match_idx = image_diffs[0][0]
        best_image, _, best_landmarks_2d, best_landmarks_3d = load_data_by_id(best_match_idx, self.video_df)
        if best_image is None:
            return None

        # Resize the match if needed
        if best_image.shape != frame.shape:
            best_image = cv2.resize(best_image, (frame.shape[1], frame.shape[0]))

        return ProcessorResult(frame=best_image, frame_idx=best_match_idx, landmarks=landmarks)
