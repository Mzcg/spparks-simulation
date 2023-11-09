#!/bin/bash

# Author: Zhaochen Gu
# Date:   2023-11-09
# Description: This script automates simulation runs with user-defined variable choices. It captures and stores both configuration files and results in a specified folder for subsequent visualization and analysis.

#number of processors used
num_processors=6


# Input File name
input_file="in.pattern"

#arry of specific values
spot_width_values=(1 15)
thickness_values=(1 7 28)

#save all the results to result folder
all_results_folder="multi_vars_results_folder"


# loop through range of values
for spot_width_val in "${spot_width_values[@]}"; do
	for thickness_val in "${thickness_values[@]}"; do
		#generate new file name based on the current value
		output_file="in_mpWidth_${spot_width_val}_thickness_${thickness_val}.pattern"
		
		#create parent and sub-folders with specific values for results saving
		output_result_folder="$all_results_folder/mpWidth_${spot_width_val}_thickness_${thickness_val}"
		mkdir -p "$output_result_folder"
		
		
		# Replace only the first occurrence of the pattern and save it to the new output file 
		sed "0,/3D_AM.dump#DUMP_OUTPUT_FILE/ s//"3D_AM_mpWidth_${spot_width_val}_thickness_${thickness_val}"/; 0,/15#SPOT_WIDTH/ s//${spot_width_val}/; 0, /14#THICKNESS/ s//${thickness_val}/" "$input_file" > "$output_result_folder/$output_file"
		
				
		# Change the working directory to the result folder
		cd "$output_result_folder" || exit
		
		# Run mpirun for the new output file (in_xxx.pattern) and save the output to the specific folder
		spk_mpi_location='/mnt/c/Users/zg0017/Documents/Fall 2023/Simulation/spparks-6Sep23/examples/am_path/pattern_repeat/spk_mpi'
		mpirun -np "$num_processors" "$spk_mpi_location" < "$output_file" 
			
		# Change back to original working directory
		cd - || exit
	
	done
done