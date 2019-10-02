import glob
from glob import glob

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from tqdm import tqdm


def overlay_landmarks_on_frame(landmarks, frame):
    """
    :param landmarks: 2d landmarks for the face
    :param frame: corresponding image
    :return: frame with face drawn on top
    """
    frame = frame.astype(np.uint8)
    landmarks = landmarks.astype(np.int32)
    # define which points need to be connected with a line
    jawPoints = [0, 17]
    rigthEyebrowPoints = [17, 22]
    leftEyebrowPoints = [22, 27]
    noseRidgePoints = [27, 31]
    noseBasePoints = [31, 36]
    rightEyePoints = [36, 42]
    leftEyePoints = [42, 48]
    outerMouthPoints = [48, 60]
    innerMouthPoints = [60, 68]

    connectedPoints = [
        rigthEyebrowPoints,
        leftEyebrowPoints,
        noseRidgePoints,
        noseBasePoints,
        rightEyePoints,
        leftEyePoints,
        outerMouthPoints,
        innerMouthPoints,
    ]

    unconnectedPoints = [jawPoints]

    for conPts in connectedPoints:
        frame = cv2.polylines(
            frame, [landmarks[conPts[0] : conPts[1]]], isClosed=True, color=[255, 255, 255], thickness=1
        )

    for conPts in unconnectedPoints:
        frame = cv2.polylines(
            frame, [landmarks[conPts[0] : conPts[1]]], isClosed=False, color=[255, 255, 255], thickness=1
        )

    return frame


def load_data(npz_filepath):
    """
    :param npz_filepath: .npz file from youtube-faces-with-keypoints dataset
    :return: colorImages, boundingBox, landmarks2D, landmarks3D
    """
    with np.load(npz_filepath) as face_landmark_data:
        colorImages = face_landmark_data["colorImages"]
        boundingBox = face_landmark_data["boundingBox"]
        landmarks2D = face_landmark_data["landmarks2D"]
        landmarks3D = face_landmark_data["landmarks3D"]

    return colorImages, boundingBox, landmarks2D, landmarks3D


def plot_images_in_row(*images, size=3, titles=None):
    """
    param: images to plot in a row
    :param size: inches size for the plot
    :param titles: subplot titles

    return: matplotlib figure
    """
    fig = plt.figure(figsize=(size * len(images), size))

    if titles is None:
        titles = ["" for _ in images]

    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.title(title)
        plt.imshow(image)

    return fig


def debug_landmark_images(npz_path):
    """
    :param npz_path: filepath to a single preprocessed video
    :return: matplotlib figure
    """
    colorImages, boundingBox, landmarks2D, landmarks3D = load_data(npz_path)
    print(list(map(lambda x: x.shape, [colorImages, boundingBox, landmarks2D, landmarks3D])))

    viz_images = []

    num_images = colorImages.shape[-1]  # 240

    for i in range(0, num_images, 40):
        img = colorImages[..., i].astype(np.uint8)
        landmarks = landmarks2D[..., i].astype(np.int32)
        img = overlay_landmarks_on_frame(landmarks, img)
        viz_images.append(img)

    return plot_images_in_row(*viz_images)


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
    image_diff_W = 0.1
    keypoint_diff_W = 0.9

    img_diff = np.mean(np.abs(cv2.resize(last_frame, current_frame.shape[:2][::-1]) - current_frame) > 25)

    # normalizing keypoints to be [0; 1]
    qshape = np.array(query_image.shape[:2])
    cshape = np.array(current_frame.shape[:2])

    keypoints_query_norm = keypoints_query / qshape
    keypoints_current_norm = keypoints_current / cshape

    keypoint_diff = np.mean(np.abs(keypoints_query_norm - keypoints_current_norm)) * 10

    return float(img_diff) * image_diff_W, float(keypoint_diff) * keypoint_diff_W


class FaceEmbeddingGenerator:
    """
    Base class of embedding generators
    """

    @staticmethod
    def make_embedding(embedding):
        return embedding.flatten()


class FaceEmbeddingGenerator2D(FaceEmbeddingGenerator):
    """ Embedding generator for 3D keypoints, which holds statically the input dimensions"""

    dim = 68 * 2


class FaceEmbeddingGenerator3D(FaceEmbeddingGenerator):
    """ Embedding generator for 2D keypoints, which holds statically the input dimensions"""

    dim = 68 * 3


def build_index(embeddingMaker, videoDF, landmarks_index_args, save=True, query_loc=0):
    landmarks_index = AnnoyIndex(*landmarks_index_args)  # Approximate search index
    face_counter = 0
    videoDF = pd.read_csv("./data/youtube_faces_with_keypoints_large.csv")
    for video_i, row in tqdm(videoDF.iterrows(), total=len(videoDF)):
        # Dont add video to the index
        if video_i == query_loc:
            continue

        db_paths = glob(
            "./data/*/{videoID}.npz".format(videoID=videoDF.loc[video_i].videoID)
        )  # To face align with this
        if len(db_paths) == 0:
            continue

        db_path = db_paths[0]
        db_colorImages, db_boundingBox, db_landmarks2D, db_landmarks3D = load_data(db_path)

        start_index = face_counter
        for frame_i in range(db_colorImages.shape[-1]):
            face_counter += 1
            landmarks_index.add_item(face_counter, embeddingMaker.make_embedding(db_landmarks2D[..., frame_i]))
        end_index = face_counter

        videoDF.at[video_i, "start"] = start_index
        videoDF.at[video_i, "end"] = end_index

    print("Building index...")
    landmarks_index.build(10)  # 10 trees

    if save:
        print("Saving index...")
        landmarks_index.save("landmarks.ann")

        print("Saving csv alongside index...")
        videoDF.to_csv("data/youtube_faces_with_keypoints_large.csv", index=False)

    return landmarks_index, videoDF


def load_index_and_metadata(landmarks_index_args):
    print("Loading metadata csv...")
    videoDF = pd.read_csv("./data/youtube_faces_with_keypoints_large.csv")

    print("Loading face embedding index...")
    landmarks_index = AnnoyIndex(*landmarks_index_args)
    landmarks_index.load("landmarks.ann")  # super fast, will just mmap the file

    return landmarks_index, videoDF


def load_data_by_id(id, videoDF):
    """Given an frame ID, and a dataset description"""
    video = videoDF[(videoDF.start < id) & (videoDF.end > id)]
    if len(video) == 0:
        return None, None, None, None

    paths = glob("./data/*/{videoID}.npz".format(videoID=video.iloc[0].videoID))
    if len(paths) == 0:
        return None, None, None, None

    path = paths[0]
    frame_id = int(id - video.start)
    q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(path)
    return (
        q_colorImages[..., frame_id],
        q_boundingBox[..., frame_id],
        q_landmarks2D[..., frame_id],
        q_landmarks3D[..., frame_id],
    )


def load_image_by_id(id, videoDF):
    """Utility function to get a single frame by ID"""
    return load_data_by_id(id, videoDF)[0]


@click.command()
# @click.argument('-i', '--input', type=click.Path(exists=True))
# @click.argument('-o', '--output', type=click.Path(exists=False))
@click.option("-idx", "--idx", type=int, default=10)
@click.option("--index", type=click.Choice(["build", "load"]), default="load")
def face_alignement(idx, index="build"):
    videoDF = pd.read_csv("./data/youtube_faces_with_keypoints_large.csv")

    # Setting up query video
    query_loc = idx
    query_path = glob("./data/*/{videoID}.npz".format(videoID=videoDF.loc[query_loc].videoID))[
        0
    ]  # To face align with this
    q_colorImages, q_boundingBox, q_landmarks2D, q_landmarks3D = load_data(query_path)

    embeddingMaker = FaceEmbeddingGenerator2D()  # Embedding generator for 3D face keypoints
    landmarks_index_args = [embeddingMaker.dim, "euclidean"]

    videoDF = pd.read_csv("./data/youtube_faces_with_keypoints_large.csv")

    if index == "load":
        landmarks_index, videoDF = load_index_and_metadata(landmarks_index_args=landmarks_index_args)
    elif index == "build":
        landmarks_index, videoDF = build_index(embeddingMaker, videoDF, landmarks_index_args)
    else:
        raise ValueError("Index didn't get built or loaded.")

    last_predicted_image = q_colorImages[..., 0]
    for i in tqdm(range(q_colorImages.shape[-1])):
        query_image = q_colorImages[..., i]
        nns, dists = landmarks_index.get_nns_by_vector(
            embeddingMaker.make_embedding(q_landmarks2D[..., i]), 10, include_distances=True
        )

        best_matches = [(image_i, dist) for image_i, dist in zip(nns, dists)]
        image_diffs = sorted(best_matches, key=lambda x: x[1], reverse=True)  # sort by distance

        best_match_idx = image_diffs[0][0]
        best_image, _, best_landmarks2D, best_landmarks3D = load_data_by_id(best_match_idx, videoDF)

        if not (last_predicted_image is None or best_image is None):
            plot_images_in_row(
                last_predicted_image,
                overlay_landmarks_on_frame(best_landmarks2D, best_image),
                overlay_landmarks_on_frame(q_landmarks2D[..., i], query_image),
                titles=["Last frame", f"Query {i}", f"Target {best_match_idx}"],
            )

        # Something went badly wrong.
        if last_predicted_image is None or best_image is None:
            print(
                f"last_predicted_image is None: {last_predicted_image is None}; best_image is None: {best_image is None}"
            )
            continue
        else:
            frame_cost = frame_cost_function(
                last_predicted_image, best_image, q_landmarks2D[..., i], best_landmarks2D, query_image
            )
            last_predicted_image = best_image

        plt.suptitle(f"Cost: {frame_cost}")
        dbg_name = f"debug/debug_{i:03d}.png"
        print(f"Saving debug image: {dbg_name}")
        plt.savefig(dbg_name)

        plt.close(fig="all")

    # TODO: Documentation
    # TODO: Command line utility that takes video in, and returns generated video


if __name__ == "__main__":
    face_alignement()
