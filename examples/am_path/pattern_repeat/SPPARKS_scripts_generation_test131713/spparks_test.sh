#This program is just a test for running commands on linux system. want to use root folder
#test on 2/1/2024 12:15pm, seems work.
#we will embed them to our Python code (parameters_sets_generation.py)

#!/bin/bash

# Set the root folder
root="/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/SPPARKS_scripts_generation_test131713"

# Set spparks_exe
spparks_exe="$root/../spk_mpi"

# Set target_folder
target_folder="$root/speed_45_mpwidth_69_haz_114.0_thickness_10.0/3D_AMsim_speed_45_mpwidth_69_haz_114.0_thickness_10.0.in"

# Print the values for verification
echo "Root folder: $root"
echo "SPPARKS executable: $spparks_exe"
echo "Target folder: $target_folder"

# You can use the variables in your further commands or operations
# For example:
timeout 10s mpirun -np 28 "$spparks_exe" < "$target_folder"