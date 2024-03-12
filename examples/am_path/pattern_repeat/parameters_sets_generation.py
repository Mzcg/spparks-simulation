import itertools
import subprocess
#import pandas as pd
import os
from pathlib import Path

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
speed_list = [3, 13, 23, 33, 42, 52, 63, 72]
mpwidth_list = [40, 35, 30, 25, 20, 15, 10]
haz_list = [91, 82, 74, 65, 56, 48, 40]
thickness_list = [11, 8, 7, 5]


#mpwidth_list = [10]
#haz_list = [40]
#speed_list = [3]
#speed_list = [3]
#mpwidth_list = [69, 44, 25]

#thickness_list = [5]
#mpwidth_list = [69, 25]


#thickness_list = [14]

print("speedlist: ", speed_list)
print("mpwidthlist: ", mpwidth_list)
print("haz:", haz_list)
print("thickness: ", thickness_list)


#step 3: generate all combinations of parameters and values
combinations = list(itertools.product(speed_list, mpwidth_list, haz_list, thickness_list))
print(len(combinations), " combinations:" , combinations)


#step 4: generate spparks commands and run
#spk_mpi_location = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\spk_mpi"
#generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"
#generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\SPPARKS_scripts_generation_test201425"
working_directory = os.path.dirname(os.path.abspath(__file__))

generated_scripts_folder_path = os.path.join(working_directory, "SPPARKS_scripts_generation_128_20240306")
command_all_filename = 'SPPARKS_commands_all.sh'
print("file_path: ", generated_scripts_folder_path)
commands_all_path = os.path.join(generated_scripts_folder_path, command_all_filename)

# Create the directory if it doesn't exist
os.makedirs(generated_scripts_folder_path, exist_ok=True)

with open(commands_all_path, 'w', newline='\n') as file: #saving spparks command to a file. #b: in binary mode(unix readable)

    #generated folder path for commands running in server/linux/cluster system.
    #line1-set root folder(no repeat): e.g.: root="/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/SPPARKS_scripts_generation"
    root_scripts_folder_command = generated_scripts_folder_path
    #root_scripts_folder_command = root_scripts_folder_command.replace('\\', '/').replace('C:/', '/mnt/c/')
    #file.write("root=\""+root_scripts_folder_command+"\""+"\n") #sample: root="/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/SPPARKS_scripts_generation_test_HPC_202"
    root_scripts_folder_command = root_scripts_folder_command.replace('\\', '/').replace('C:/Users/zg0017/PycharmProjects', '/work/08207/zg0017') #for hpc path:
    file.write("root=\""+root_scripts_folder_command+"\""+"\n") #sample: root=

    #line2- set spk_mpi location(no repeat):
    spk_mpi_location_command = "spk_mpi_location=\"$root/../spk_mpi\""
    file.write(spk_mpi_location_command+"\n")

    #create command  line arguments
    for comb in combinations:
        command_line_args = f"-speed {comb[0]} -mpwidth {comb[1]} -haz {comb[2]} -thickness {comb[3]}"
        script_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\command_line_arg.sh "
        #script_path = "/mnt/c/Users/zg0017/PycharmProjects/spparks-simulation/examples/am_path/pattern_repeat/command_line_arg.sh " #linux path (this is for test now)
        command = script_path + command_line_args

        print("command: ", command)

        # Set the working directory to the parent directory of the new_scripts_folder
        #working_directory = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat"
        # run the command line to active running shell script for generating new SPPARKS script files into separate folder (code in command_line_arg.sh)
        #working_directory = os.getcwd() #e.g: r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat"
        result =subprocess.run(command, check=True, shell=True, cwd=working_directory) #need to set the working directory to generate new folder using command_line_arg.sh file
        result.check_returncode()  # Raise an exception if the command fails

        # Generate SPPARKS running command for multiple combination of variables
        #spk_mpi_location = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\spk_mpi"
        #spk_mpi_location = spk_mpi_location.replace('\\', '/')
        #generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"
        generated_script_subfolder_name = "speed_"+str(comb[0])+"_mpwidth_" + str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])
        script_file_name = "3D_AMsim_speed_"+str(comb[0])+"_mpwidth_"+str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])+".in"
        script_file_path = os.path.join(generated_scripts_folder_path, generated_script_subfolder_name, script_file_name)
        dump_output_path = os.path.join(generated_scripts_folder_path, generated_script_subfolder_name)
        #path format adjustment (backslash to forward slash)
        script_file_path = script_file_path.replace('\\', '/')
        dump_output_path = dump_output_path.replace('\\', '/')

        # target_script_file_command = "target_spparks_script=\"$root/"+generated_script_subfolder_name+"/"+script_file_name+"\""
        target_script_file_command = "target_script_file=\"$root/" + generated_script_subfolder_name + "/" + script_file_name + "\""
        file.write(target_script_file_command + "\n")

        dump_output_path_command = "dump_output_folder=\"$root/" + generated_script_subfolder_name + "/" + "\""
        file.write(dump_output_path_command + '\n')

        #write all the running commands to a file
        #command line 1: start program and report time
        command_start_line = "echo \"Start Program : $(date)\" >> logAllruns.log"  #log file name: logAllruns.log
        #unix_command_start_line = command_start_line.replace('\r\n', '\n')

        #command line 2: mpirun (run sppakrs scripts)
        num_processors = 28
        time_limit = "timeout 10s "
        #sppark_command = time_limit + "mpirun -np "+str(num_processors)+ " " + "\""+spk_mpi_location + "\""+" < " + "\""+script_file_path + ".in\" >> ShortRunSpparks.log"
        #sppark_command = time_limit + "mpirun -np "+str(num_processors)+ " " + "\"$spk_mpi_location\"" + " < " + "\"target_spparks_script\"" + " >> ShortRunSpparks.log"
        ### option 1: command with time limit:
        #sppark_command = time_limit + "mpirun -np "+str(num_processors)+ " " + "\"$spk_mpi_location\"" + " < " + "\"$target_script_file\"" + " >> ShortRunSpparks.log"
        #option 2: command without time limit
        #sppark_command = "mpirun -np "+str(num_processors)+ " " + "\"$spk_mpi_location\"" + " < " + "\"$target_script_file\"" + " >> ShortRunSpparks.log"
        #option 2 (TACC hpc version)
        sppark_command = "ibrun " + "\"$spk_mpi_location\"" + " < " + "\"$target_script_file\"" + " >> DetailsRunSpparks.log"
        sppark_command = sppark_command.replace('C:/', '/mnt/c/')  #if use the string, this line may not userful anymore, but leave it here not changing anything. for time being, it's fine.
        #unix_sppark_command = sppark_command.replace('\r\n', '\n')
        print("SPPARKS COMMAND: ", sppark_command)

        #command line 3: move simulated dump files to corresponding folder

        #move_dump_file = "mv ./*.dump " + dump_output_path
        move_dump_file_command = "mv ./*.dump " + "\"$dump_output_folder\""
        move_dump_file_command = move_dump_file_command.replace('C:/', '/mnt/c/')
        print("move dump file: ", move_dump_file_command)

        #command line 4: move simulated log.spparks file to corresponding folder
        #move_log_file = "mv ./*.spparks " + dump_output_path
        move_log_file_command = "mv ./*.spparks " + "\"$dump_output_folder\""
        move_log_file_command = move_log_file_command.replace('C:/', '/mnt/c/')
        #unix_move_log_file = move_log_file.replace('\r\n', '\n')
        print("move log.spparks file: ", move_log_file_command)

        # command line 5: move simulated tmp.lammps.variable file to corresponding folder
        #move_lammps_file = "mv ./*.variable " + dump_output_path
        move_lammps_file_command = "mv ./*.variable " + "\"$dump_output_folder\""
        move_lammps_file_command = move_lammps_file_command.replace('C:/', '/mnt/c/')
        #unix_move_lammps_file = move_lammps_file.replace('\r\n', '\n')
        print("move tmp.lammps.variable file: ", move_lammps_file_command)

        #command line 6: print which folder we are working on now for writing to running log
        command_folder_line = "echo \"prgram ran for folder "+ generated_script_subfolder_name+"\" >> logAllruns.log"
        #unix_command_folder_line = command_folder_line.replace('\r\n', '\n')

        #command line 7: end command for one simulation.
        command_end_line = "echo \"End Program : $(date)\" >> logAllruns.log"  # log file name: logAllruns.log
        #unix_command_end_line = command_end_line.replace('\r\n', '\n')

        #write commands to the file "SPPARKS_commands_all.txt"
        file.write(command_start_line + '\n')
        file.write(sppark_command + '\n')
        file.write(move_dump_file_command + '\n')
        file.write(move_log_file_command + '\n')
        file.write(move_lammps_file_command + '\n')
        file.write(command_folder_line + '\n')
        file.write(command_end_line + '\n')













