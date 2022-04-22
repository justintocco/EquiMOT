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
        #breakpoint()
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
                #breakpoint(
                img = int(str(img)[:-4])
                img = int(img * 30.0 / ffmpeg_fps)
                #breakpoint()
                number = set_name[3:5]
                #breakpoint()
                old_dest = video_path + orig_img
                new_dest = "/s{}_vid00{}_f".format(number, naming[number])
                os.rename(video_path + "/" + orig_img, new_dir + new_dest + str(img) +".png")

organize()

    # dir_path = "PIE/images/set03/video_0004"
    # directory = os.listdir(dir_path)
    # new_dir = "processed_images"
    # cfg = open("config.json")
    # config = json.load(cfg)
    # cfg.close()
    # #breakpoint()
    # #copy/move each of them to a new folder inside of EquiMOT "processed_images"
    # for img in directory:
    #     print(img)
    #     orig_img = img
    #     img = int(str(img)[:-4])
    #     img = img * 30.0 / config['ffmpeg_fps']
    #     #breakpoint()
    #     os.rename(dir_path +"/" + orig_img, new_dir + "/s03_vid0004_f" + str(img) +".png")
    # #rename each of them using frame rate found in config.json and using the set and video id
    #     #example: 00010.png from set01/vid_0001 becomes s01_vid_0001_f(00010*30fps / FR).png