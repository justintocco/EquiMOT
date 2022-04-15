from operator import mod
import PIE.pie_data as pie
import os

def main():
    db = pie.PIE(regen_database=False, data_path='./PIE')
    db.generate_database()
    #db.get_data_stats()
    #frame_nums1 = db.get_frame_numbers('set01')
    #print(frame_nums1)
    ann_frames1 = db.get_annotated_frame_numbers('set01')
    print(ann_frames1['video_0001'])
    print(os.environ["FFMPEG_FRAME_RATE"])


if __name__ == "__main__":
    main()
