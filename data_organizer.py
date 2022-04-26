import os
import json

def organize():
    #iterate across the directories in the ./PIE/images/set0*/videos_****
    
    set_path = "PIE/images/"
    sets = os.listdir(set_path)
    new_dir = "processed_images"
    cfg = open("config.json")
    config = json.load(cfg)
    cfg.close()
    ffmpeg_fps = config['ffmpeg_fps']
    naming = {
        "01": "03",
        "02": "02",
        "03": "03",
        "04": "12",
        "05": "01"
    }
    for set_ in sets:
        if set_.startswith('.DS_Store'):
            continue
        path = set_path + set_ + "/"
        video_dir = os.listdir(path)
        set_name = set_
        for video in video_dir:
            if video.startswith('.DS_Store'):
                continue
            video_path = path + video + "/"
            imgs = os.listdir(video_path)
            if len(imgs) == 0:
                continue
            for img in imgs:
                orig_img = img
                img = int(str(img)[:-4])
                img = int(img * 30.0 / ffmpeg_fps)
                number = set_name[3:5]
                old_dest = video_path + orig_img
                new_dest = "/s{}_vid00{}_f".format(number, naming[number])
                os.rename(video_path + "/" + orig_img, new_dir + new_dest + str(img) +".png")

organize()