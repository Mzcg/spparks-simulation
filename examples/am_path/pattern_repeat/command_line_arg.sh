#!/bin/bash
#sample command line input: ./command_line_arg.sh -speed 1 -mpwidth 2 -thickness 3 -haz 4

while [[ $# -gt 0 ]]; do
  case "$1" in
    -speed)
      speed="$2"
      shift 2 # move to the next set of two arguments
      ;;
    -thickness)
      thickness="$2"
      shift 2
      ;;
    -mpwidth)
      mpwidth="$2"
      shift 2
      ;;
    -haz)
      haz="$2"
      shift 2;
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

#set frame_time_gap based on speed
case "$speed" in
  3)
    frame_time_gap=3000
    ;;
  9)
    frame_time_gap=1000
    ;;
  15)
    frame_time_gap=600
    ;;
  21)
    frame_time_gap=450
    ;;
  27)
    frame_time_gap=350
    ;;
  33)
    frame_time_gap=300
    ;;
  39)
    frame_time_gap=250
    ;;
  45)
    frame_time_gap=200
    ;;
  *)
    echo "Invalid speed value: $speed"
    exit 1
    ;;
esac

#echo "Speed: $speed, Power: $power, mpwidth: $mpwidth, haz: $haz"

### Send self-defined parameters from command line to the SPPARKS script
#get original input file: SPPARKS script
input_file="in.pattern"

#name new folder for saving NEW scripts (modified parameter values)
#new_scripts_folder='generated_SPPARKS_scripts'
#new_scripts_folder='SPPARKS_scripts_generation_test_HPC_box256_18'
new_scripts_folder='SPPARKS_scripts_generation_splittest_20240223'


#create parent and sub-folders with specific values for results saving
output_scripts_folder="$new_scripts_folder/speed_${speed}_mpwidth_${mpwidth}_haz_${haz}_thickness_${thickness}"

# Check if the folder exists,, if exists no need to add folder
if [ -d "$output_scripts_folder" ]; then
  echo "Folder already exists: $output_scripts_folder"
else
  # Create the folder
  mkdir -p "$output_scripts_folder"
  echo "Folder created: $output_scripts_folder"
fi

#step 2: modify script files with new values obtained from command line
output_script="3D_AMsim_speed_${speed}_mpwidth_${mpwidth}_haz_${haz}_thickness_${thickness}.in"

#create parent and sub-folders with specific values for results saving
#output_scripts_folder="$new_scripts_folder/speed_${speed}_thickness_${thickness}_mpwidth_${mpwidth}_haz_${haz}"
#mkdir -p "$output_scripts_folder"

# Replace only the first occurrence of the pattern and save it to the new output file
#sed "0,/3D_AM_*.dump#DUMP_OUTPUT_FILE/ s//"3D_AM_speed_${speed}_mpwidth_${mpwidth}_haz_${haz}_thickness_${thickness}_*.dump"/; 0,/10.0#SPEED/ s//${speed}/; 0, /15#SPOT_WIDTH/ s//${mpwidth}/; 0,/25#HAZ/ s//${haz}/; 0, /14#THICKNESS/ s//${thickness}/;"  "$input_file" > "$output_scripts_folder/$output_script"
#sed "0,/10.0#SPEED/ s//${speed}/; 0, /15#SPOT_WIDTH/ s//${mpwidth}/; 0,/25#HAZ/ s//${haz}/; 0, /14#THICKNESS/ s//${thickness}/; 0, /50#FRAME_TIME_GAP/ s//${frame_time_gap}/;"  "$input_file" > "$output_scripts_folder/$output_script"
sed "0,/10.0#SPEED/ s//${speed}/; 0, /15#SPOT_WIDTH/ s//${mpwidth}/; 0,/25#HAZ/ s//${haz}/; 0, /14#THICKNESS/ s//${thickness}/; 0, /50#FRAME_TIME_GAP/ s//${frame_time_gap}/;"  "$input_file" > "$output_scripts_folder/$output_script"

