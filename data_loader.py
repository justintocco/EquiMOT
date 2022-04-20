import PIE.pie_data as pie
import pickle
import json
from os.path import join, isfile

def main():
    model = pie.PIE(regen_database=False, data_path='./PIE')
    db = model.generate_database()
    # #db.get_data_stats()
    # #frame_nums1 = db.get_frame_numbers('set01')
    # #print(frame_nums1)
    # ann_frames1 = model.get_annotated_frame_numbers('set01')
    # #print(ann_frames1)
    # print("SET 1")
    # for vid in ann_frames1:
    #     print(ann_frames1[vid][0])

    # ann_frames2 = model.get_annotated_frame_numbers('set02')
    # print("SET 2")
    # for vid in ann_frames2:
    #     print(ann_frames2[vid][0])
    
    # ann_frames3 = model.get_annotated_frame_numbers('set03')
    # print("SET 3")
    # for vid in ann_frames3:
    #     print(ann_frames3[vid][0])

    small_db = {}
    for set_id in db:
        small_db[set_id] = {}
        for vid_id in db[set_id]:
            small_db[set_id][vid_id] = {}
            for obj_id in db[set_id][vid_id]["traffic_annotations"]:
                obj = db[set_id][vid_id]["traffic_annotations"][obj_id]
                for i, frame in enumerate(obj['frames']):
                    if frame not in small_db[set_id][vid_id]:
                        small_db[set_id][vid_id][frame] = []
                    small_db[set_id][vid_id][frame].append({
                        'bbox': obj['bbox'][i],
                        'class': obj['obj_class'],
                        'uid': obj_id
                    })

    #print(json.dumps(small_db, indent=4))
    cache_file = 'small_database.pkl'
    if isfile(cache_file):
        print('cache file already exists at {}'.format(cache_file))
        return

    with open(cache_file, 'wb') as fid:
        pickle.dump(small_db, fid, pickle.HIGHEST_PROTOCOL)
    print('The EquiMOT database is successfully written to {}'.format(cache_file))

    #print(db['set01']['video_0001']['traffic_annotations'])


if __name__ == "__main__":
    main()
