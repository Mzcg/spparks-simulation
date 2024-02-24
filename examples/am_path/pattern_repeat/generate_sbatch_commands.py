import os

current_folder = os.getcwd()
#current_folder = "SPPARKS_scripts_generation_splittest_20240223"
get_all_files = os.listdir(current_folder)
slurm_file_count = sum(1 for file in get_all_files if file.startswith("slurm"))

print(slurm_file_count) #e.g = 4 (4 slurm-split_file_x)

with open("sbatch_commands_launch.sh", "w") as file:
    for slurm_file_index in range (1, slurm_file_count+1):
        print(slurm_file_index)
        slurm_filename = "slurm-split_file_" +str(slurm_file_index) +".sh" #slurm-split_file_1
        command = "sbatch " + slurm_filename
        file.write(command + "\n")


