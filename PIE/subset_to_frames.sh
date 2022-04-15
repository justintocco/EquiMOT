#If you don't have ffmpeg installed on your system
#sudo apt-get install ffmpeg

CLIPS_DIR=PIE_clips             #path to the directory with mp4 videos
FRAMES_DIR=images               #path to the directory for frames

################################################################
#Ensure required programs are installed on system
if ! command -v ffmpeg &> /dev/null
then
    echo "ffmpeg could not be found. Run 'sudo apt-get install ffmpeg' then retry."
    exit
fi

if ! command -v jq &> /dev/null
then
    echo "jq could not be found. Run 'sudo apt-get install jq' then retry."
    exit
fi

curr_dir=$(pwd)
if [[ $curr_dir != *"/PIE" ]]; then
  echo "Error: Must be within PIE directory!";
  exit
fi

#Get framerate from config file
fps=$(jq .ffmpeg_fps ../config.json)

for set_dir in set01 #set02 set03 set04 set05 set06
do
    for video in ${CLIPS_DIR}/${set_dir}/*
    do
        filename=$(basename "$video")
        fname="${filename%.*}"

        #create a directory for each frame sequence
        mkdir -p ${FRAMES_DIR}/${set_dir}/$fname
        #FFMPEG will overwrite any existing images in that directory
        ffmpeg  -y -i $video -loglevel 40 -r $fps -start_number 1 -f image2 -qscale 1 ${FRAMES_DIR}/${set_dir}/$fname/%05d.png
        
    done
done