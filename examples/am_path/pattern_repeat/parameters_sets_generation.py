import itertools
import subprocess
import pandas as pd
import os

#If reading from csv files in outside folder, do following:
#Read CSV files to get values for parameters
#parameter_file_path = r"../data/parameter_selection.csv"
#data = pd.read_csv(parameter_file_path)

#convert dataframe to lists
#speed_list = data['speed'].dropna().tolist()
#mpwidth_list = data['mpwidth'].dropna().tolist()
#haz_list = data['haz'].dropna().tolist()
#thickness_list = data['thickness'].dropna().tolist()

#set up parameters selections in lists
speed_list = [3, 30, 45]
mpwidth_list = [69, 44, 25]
haz_list = [114.0, 20.0]
thickness_list = [10.0]

print("speedlist: ", speed_list)
print("mpwidthlist: ", mpwidth_list)
print("haz:", haz_list)
print("thickness: ", thickness_list)


#step 3: generate all combinations of parameters and values
combinations = list(itertools.product(speed_list, mpwidth_list, haz_list, thickness_list))
print("combinations:" , combinations)


#step 4: generate spparks commands and run
spk_mpi_location = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\spk_mpi"
generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"
command_all_filename = 'SPPARKS_commands_all.sh'
commands_all_path = os.path.join(generated_scripts_folder_path, command_all_filename)

# Create the directory if it doesn't exist
os.makedirs(generated_scripts_folder_path, exist_ok=True)

with open(commands_all_path, 'w') as file: #saving spparks command to a file.



    #create command  line arguments
    for comb in combinations:
        command_line_args = f"-speed {comb[0]} -mpwidth {comb[1]} -haz {comb[2]} -thickness {comb[3]}"
        script_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\command_line_arg.sh "
        command = script_path + command_line_args

        print(command)

        # Set the working directory to the parent directory of the new_scripts_folder
        working_directory = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat"
        # run the command line to active running shell script for generating new SPPARKS script files into separate folder (code in command_line_arg.sh)
        result = subprocess.run(command, check=True, shell=True, cwd=working_directory) #need to set the working directory to generate new folder using command_line_arg.sh file
        result.check_returncode()  # Raise an exception if the command fails

        # Generate SPPARKS running command for multiple combination of variables
        spk_mpi_location = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\spk_mpi"
        spk_mpi_location = spk_mpi_location.replace('\\', '/')
        #generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"
        generated_script_subfolder_name = "speed_"+str(comb[0])+"_mpwidth_" + str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])
        script_file_name = "3D_AMsim_speed_"+str(comb[0])+"_mpwidth_"+str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])
        script_file_path = os.path.join(generated_scripts_folder_path, generated_script_subfolder_name, script_file_name)
        dump_output_path = os.path.join(generated_scripts_folder_path, generated_script_subfolder_name)
        #path format adjustment (backslash to forward slash)
        script_file_path = script_file_path.replace('\\', '/')
        dump_output_path = dump_output_path.replace('\\', '/')


        #write all the running commands to a file
        #command line 1: start program and report time
        command_start_line = "echo \"Start Program : $(date)\" >> logAllruns.log"  #log file name: logAllruns.log

        #command line 2: mpirun (run sppakrs scripts)
        num_processors = 28
        sppark_command = "mpirun -np "+str(num_processors)+ " " + "\""+spk_mpi_location + "\""+" < " + "\""+script_file_path + ".in\" "
        sppark_command = sppark_command.replace('C:/', '/mnt/c/')
        print("SPPARKS COMMAND: ", sppark_command)

        #command line 3: move simulated dump files to corresponding folder
        move_dump_file = "mv ./*.dump " + dump_output_path
        move_dump_file = move_dump_file.replace('C:/', '/mnt/c/')
        print("move dump file: ", move_dump_file)

        #command line 4: move simulated log.spparks file to corresponding folder
        move_log_file = "mv ./*.spparks " + dump_output_path
        move_log_file = move_dump_file.replace('C:/', '/mnt/c/')
        print("move log.spparks file: ", move_log_file)

        # command line 5: move simulated tmp.lammps.variable file to corresponding folder
        move_lammps_file = "mv ./*.variable " + dump_output_path
        move_lammps_file = move_dump_file.replace('C:/', '/mnt/c/')
        print("move tmp.lammps.variable file: ", move_lammps_file)

        #command line 6: print which folder we are working on now for writing to running log
        command_folder_line = "echo \"prgram ran for folder "+ generated_script_subfolder_name+"\" >> logAllruns.log"

        #command line 7: end command for one simulation.
        command_end_line = "echo \"End Program : $(date)\" >> logAllruns.log"  # log file name: logAllruns.log

        #write commands to the file "SPPARKS_commands_all.txt"
        file.write(command_start_line + '\n')
        file.write(sppark_command + '\n')
        file.write(move_dump_file + '\n')
        file.write(move_log_file + '\n')
        file.write(move_lammps_file + '\n')
        file.write(command_folder_line + '\n')
        file.write(command_end_line + '\n')











