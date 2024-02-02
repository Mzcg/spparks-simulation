import os

current_directory = os.getcwd() #get path (e.g: C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\SPPARKS_scripts_generation_test131713)

spk_mpi_location = os.path.join(current_directory, os.pardir, "spk_mpi")
spk_mpi_location = spk_mpi_location.replace('\\','/').replace('C:/', '/mnt/c/' )

print(current_directory)
print(spk_mpi_location)

generated_script_subfolder_name = "speed_45_mpwidth_69_haz_114.0_thickness_10.0"
generated_scripts_folder_path = os.path.join(os.getcwd())
script_file_path = os.path.join(os.getcwd(), generated_script_subfolder_name, "3D_AMsim_speed_45_mpwidth_69_haz_114.0_thickness_10.0.in")
scripts_file_name= "3D_AMsim_speed_45_mpwidth_69_haz_114.0_thickness_10.0.in"
target_script_file_command = "\"$root/"+generated_script_subfolder_name+"/"+scripts_file_name+"\""
print("target_script_file_command: ",target_script_file_command)

spk_mpi_location_command = "spk_mpi_location=\"$root/../spk_mpi\""
print(spk_mpi_location_command)