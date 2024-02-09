#!/bin/bash

# Input script file
input_script="SPPARKS_commands_all.sh"

# Number of tests per file
tests_per_file=2

# Counter for file index
file_index=1

# Initialize the output file
output_file="split_file_${file_index}.sh"
log_file="logAllrun_${file_index}.log"
echo "Setting up $output_file"
echo "Setting up $log_file"
# Function to generate comprehensive header

add_header() {
    echo '#!/bin/bash'  # Shebang line
    echo ''
    echo '#SBATCH -J sim_test        # Job name'
    echo '#SBATCH -o myjob.o%j       # Name of stdout output file'
    echo '#SBATCH -e myjob.e%j       # Name of stderr error file'
    echo '#SBATCH -p development     # Queue (partition) name'
    echo '#SBATCH -N 1               # Total # of nodes'
    echo '#SBATCH -n 128             # Total # of mpi tasks'
    echo '#SBATCH -t 01:50:00        # Run time (hh:mm:ss)'
    echo '#SBATCH --mail-type=all    # Send email at begin and end of job'
    echo '#SBATCH --mail-user=ZhaochenGu@my.unt.edu'
    echo ''
    echo '# Any other commands must follow all #SBATCH directives...'
    echo 'module list'
    echo 'pwd'
    echo 'date'
    echo ''
    #echo 'root="/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/SPPARKS_scripts_generation_test_HPC_202"'
    #echo 'spk_mpi_location="$root/../spk_mpi"'

}

# Initialize flags to check if root and spk_mpi_location have been added
root_added=false
spk_mpi_added=false
# Function to reset the output file
reset_output_file() {
    output_file="split_file_${file_index}.sh"
    log_file="logRuns_${file_index}.log"
    echo "Setting up $output_file"
    echo "Setting up $log_file"
    add_header >> "$output_file"  # Call the function to generate the comprehensive header


    echo 'root="/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/SPPARKS_scripts_generation_test_HPC_202"' >> "$output_file"
    echo 'spk_mpi_location="$root/../spk_mpi"' >> "$output_file"

}


# Function to add the "End Program" line if it's missing
add_end_program() {
    end_program_line='echo "End Program : $(date)" >> '"$log_file"
    grep -q "$end_program_line" "$output_file" || echo "$end_program_line" >> "$output_file"
}

# Process the input script
count=0
#reset_output_file
add_header >> "$output_file"
while IFS= read -r line; do

    if [[ $line == "target_script_file="* ]]; then
        # Found the start of a new test
        if [ $count -eq $tests_per_file ]; then
            # Move to the next output file
            ((file_index++))
            reset_output_file
            count=0
        fi
        # Add target_script_file to the current test
        echo "$line" >> "$output_file"

        ((count++))
    elif [[ $line == "echo \"End Program:"* ]]; then
        # Found the end of a test
        echo "$line" >> "$output_file"
        # Add the "End Program" line if it's missing
        add_end_program

    else
        # Other lines, add to the current test
        echo "$line" >> "$output_file"
    fi
done < "$input_script"
# Add the "End Program" line to the last file
add_end_program

# Replace logAllruns.log with logAllrun_index.log in all split files
for i in $(seq 1 $file_index); do
    sed -i 's|logAllruns.log|logRuns_'"$i"'.log|' split_file_"$i".sh
    sed -i 's|ShortRunSpparks.log|Spparks_screen_log_'"$i"'.log|' split_file_"$i".sh
done