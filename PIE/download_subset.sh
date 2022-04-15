# Download PIE clips

curr_dir=$(pwd)

if [[ $curr_dir != *"/PIE" ]]; then
  echo "Error: Must be within PIE directory!";
  exit
fi
echo "Searching for clips directory..."

if [ ! -d ./PIE_clips ]; then
  echo "clips directory not found. Creating...";
  mkdir -p ./PIE_clips;
  mkdir -p ./PIE_clips/set01;
  mkdir -p ./PIE_clips/set02;
  mkdir -p ./PIE_clips/set03;
  mkdir -p ./PIE_clips/set04;
  mkdir -p ./PIE_clips/set05;
  mkdir -p ./PIE_clips/set06;
else
  echo "Error: clips directory already exists. Remove directory and retry."
  exit
fi

wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set01/video_0001.mp4 ./PIE_clips/set01
wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set02/video_0001.mp4 ./PIE_clips/set02
wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set03/video_0004.mp4 ./PIE_clips/set03
wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set04/video_0005.mp4 ./PIE_clips/set04
wget -N --recursive --no-parent -nH --cut-dirs=1 -R "index.html*" https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set05/video_0001.mp4 ./PIE_clips/set05

echo "Download successful."