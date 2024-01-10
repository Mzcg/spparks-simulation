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

#echo "Speed: $speed, Power: $power, mpwidth: $mpwidth, haz: $haz"

### Send self-defined parameters from command line to the SPPARKS script
#get original input file: SPPARKS script
input_file="in.pattern"

#name new folder for saving NEW scripts (modified parameter values)
new_scripts_folder='generated_SPPARKS_scripts'

#step 2: modify script files with new values obtained from command line
output_script="3D_AMsim_speed_${speed}_thickness_${thickness}_mpwidth_${mpwidth}_haz_${haz}.in"

#create parent and sub-folders with specific values for results saving
output_scripts_folder="$new_scripts_folder/speed_${speed}_thickness_${thickness}_mpwidth_${mpwidth}_haz_${haz}"
mkdir -p "$output_scripts_folder"

# Replace only the first occurrence of the pattern and save it to the new output file
sed "0,/3D_AM_*.dump#DUMP_OUTPUT_FILE/ s//"3D_AM_speed_${speed}_thickness_${thickness}_mpwidth_${mpwidth}_haz_${haz}_*.dump"/; 0,/10.0#SPEED/ s//${speed}/; 0, /15#SPOT_WIDTH/ s//${mpwidth}/; 0,/25#HAZ/ s//${haz}/; 0, /14#THICKNESS/ s//${thickness}/;"  "$input_file" > "$output_scripts_folder/$output_script"
