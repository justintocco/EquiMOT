# EquiMOT

Authors: Justin Tocco, Jacob Slimak, Carlos Cardenas, Jett Li

**Download links for videos:** [YorkU server](http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/) or [Dropbox](https://www.dropbox.com/sh/1th9hjcrce8sof1/AADKIF9itB7KmRvgH4iQxvCpa?dl=0)

### Table of contents
* [Project Overview](#overview)
* [Downloading Videos/Images](#download)
* [Generate .pkl Annotation Dictionary](#generate)

<a name="overview"></a>
# Project Overview
EECS 442 final project.

<a name="download"></a>
# Downloading Videos/Images
Training and validation data are a subset of the PIE (Pedestrian Intention Estimation) dataset.

Specifically, the following 5 videos are used:

* set01
  * video_0001.mp4
* set02
  * video_0001.mp4
* set03
  * video_0004.mp4
* set04
  * video_0005.mp4
* set05
  * video_0001.mp4

To download this subset of videos, run script `download_subset.sh` from _within the PIE directory_.

To extract frames from the videos, use `subset_to_frames.sh`. Each video is 30fps by default. The `ffmpeg_fps` parameter in `config.json` modifies how often to extract frames (be careful about PIE frame_id scheme being out of sync with image naming scheme).

<a name="generate"></a>
# Generate .pkl Annotation Dictionary
Run `data_loader.py` to generate the EquiMOT-compatible version of the PIE database. The `.pkl` dictionary file will be cached and should only need to be generated once.
