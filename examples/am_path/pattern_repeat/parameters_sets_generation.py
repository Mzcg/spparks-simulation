import itertools
import subprocess
import pandas as pd
import os


#step 1: Read CSV files to get values for parameters
parameter_file_path = r"../data/parameter_selection.csv"
data = pd.read_csv(parameter_file_path)

#step 2: convert dataframe to lists
speed_list = data['speed'].dropna().tolist()
mpwidth_list = data['mpwidth'].dropna().tolist()
haz_list = data['haz'].dropna().tolist()
thickness_list = data['thickness'].dropna().tolist()

print("speedlist: ", speed_list)
print("mpwidthlist: ", mpwidth_list)
print("haz:", haz_list)
print("thickness: ", thickness_list)


#step 3: generate all combinations of parameters and values
combinations = list(itertools.product(speed_list, mpwidth_list, haz_list, thickness_list))
print("combinations:" , combinations)

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

    generated_scripts_folder_path = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"
    generated_script_subfolder_name = "speed_"+str(comb[0])+"_mpwidth_" + str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])
    script_file_name = "3D_AMsim_speed_"+str(comb[0])+"_mpwidth_"+str(comb[1])+"_haz_"+str(comb[2])+"_thickness_"+str(comb[3])
    script_file_path = os.path.join(generated_scripts_folder_path, generated_script_subfolder_name, script_file_name)
    dump_output_path = os.path.join(generated_scripts_folder_path, generated_scripts_folder_path)

    num_processors = 28
    sppark_command = "mpirun -np -"+str(num_processors)+ " " + "\""+spk_mpi_location + "\""+" < " + "\""+script_file_path + ".in" " > " + "\"" + dump_output_path + "\""

    print("SPPARKS COMMAND: " ,sppark_command)




#generate mpirun commands for running SPPARKS
#







