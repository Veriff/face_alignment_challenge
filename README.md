# Face alignment challenge

**There is a grand prize of 500€ for the first place and additional smaller prizes for others!**

## Goal

#### The goal of the challenge is to make a good-looking montage of images from videos from a database of celebrity videos to match an unknown target video! We will also look through best scoring videos and add points for creativity!

Good-looking face-alignment has been defined as a cost function that 
incentivizes keeping visual differences low between consecutive frames
and faces aligned between target and produced outputs.

For creativity, do image warping, have videos of only a single celebrity, add a beat, do face swaps...
let your creativity flow and make us laugh! :)

Please send your solutions to challenge@veriff.com, more detailed format under [Submitting](https://github.com/Jonksar/face_alignement_challenge#submitting).

## Constraints

1. Processing all 240 frames in a single video should not take more than 1 minute. 
2. Building an index should not take more than 10 minutes.
3. We will be running your solution in a standard p2.xlarge machine in AWS (CPU, GPU and memory constraints come from there).

## Timeline

1. Challenge was made for Pycon Estonia 2019 on October 3rd, 2019.
2. Participations after October 10th, will not be considered valid.
3. Veriff will announce the winners on October 17th.

## Running the code
You will need Python 3.6 or later.

### Setup

In order to get started, clone current github repository for solution interface
and set up Python development environment:

```
git clone git@github.com:Veriff/face_alignment_challenge.git
cd face_alignment_challenge

# You can set up an virtual environment here
python3 -m venv venv
. venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Download the data
We recommend using a `data/` directory within the root of this repositiory.

Alternative to using wget is to use AWS cli:
```
aws s3 cp s3://veriff-face-alignment-challenge/FILENAME .
```
#### Download only 10% of the data, to get started faster ~1GB
```
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/small.zip
```

Contents:
```
train
├── Alison_Lohman_0.npz
├── ...
├── youtube_faces_with_keypoints_small.csv

test
├── Vicente_Fox_1.npz
├── ...
├── youtube_faces_with_keypoints_small.csv
```

#### Download the remaining data ~10GB
```
wget https://veriff-face-alignment-challenge.s3-eu-west-1.amazonaws.com/large.zip
```

Contents:
```
large_train
├── Abdel_Aziz_Al-Hakim_0.npz
├── ...
├── youtube_faces_with_keypoints_large.csv
```

Move content of this directory to `train/`: 
```
mv large_train/* train
```

### Using command line interface:

You can get running with:
```
# Help about the command line interface
python cli.py --help

# Build file index, takes about 20s
python cli.py index --videos data/train/youtube_faces_with_keypoints_small.csv

# Process a video, matching it against the index.
python cli.py process-video -v PATH_TO_VIDEO_NPZ
```

We also provide a baseline model that you can try by adding `--baseline` flag after `cli.py`:

    python cli.py --baseline index 
    python cli.py --baseline process-video -v PATH_TO_VIDEO


## Participating

In `processor.py`, you can find a Processor class that is abstraction for your solution.
Read through the comments and documentation in that class and add your own solution.

You can also find a baseline approach on solving the problem as BaselineProcessor,
that does OK in face-alignment, but does not work well for frame difference part of the cost function.
It is OK to use baseline approach as a starting point and improve upon it.


## Submitting
Please send your solutions to challenge@veriff.com as an attachement.

Please also add the following details in the e-mail as well: 
```
Name:
Telephone number:
Short description:
Feedback about challenge (what was fun, what was frustrating etc.):
```
