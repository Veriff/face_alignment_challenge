import logging

import click

from baseline import BaselineProcessor
from processor import Processor
from utils.logging import setup_logging

logger = logging.getLogger(__name__)
videos_csv_filename = "./data/youtube_faces_with_keypoints_large.csv"

processor_cls = Processor


@click.group()
@click.option("--debug", "-d", is_flag=True)
@click.option("--baseline", "-b", is_flag=True)
def cli(debug: bool, baseline: bool):
    global processor_cls

    setup_logging(debug=debug)

    if baseline:
        processor_cls = BaselineProcessor


@cli.command()
@click.option("--videos", "-v", type=click.Path(exists=True, dir_okay=False), default=videos_csv_filename)
def index(videos: str):
    logger.info("Creating %s", processor_cls.__name__)
    processor = processor_cls(videos)

    logger.info("Building index")
    index_filename = f"{videos}.index"
    processor.build_index(index_filename)


@cli.command()
@click.option("--videos", "-v", type=click.Path(exists=True), default=videos_csv_filename)
@click.argument("filename", type=click.Path(exists=True, dir_okay=False))
def process_video(videos: str, filename: str = None):
    logger.info("Creating %s", processor_cls.__name__)
    processor = processor_cls(videos)

    index_filename = f"{videos}.index"
    processor.load_index(index_filename)
    logger.info("Processing video from %s", filename)
    processor.process_video(filename)


@cli.command()
@click.option("--cuda", "-c", is_flag=True)
@click.option("--videos", "-v", type=click.Path(exists=True), default=videos_csv_filename)
def process_webcam(cuda: bool, videos: str):
    logger.info("Creating %s", processor_cls.__name__)
    processor = processor_cls(videos)

    index_filename = f"{videos}.index"
    processor.load_index(index_filename)
    logger.info("Processing video from webcam")
    processor.process_webcam(use_cuda=cuda)


if __name__ == "__main__":
    cli()
