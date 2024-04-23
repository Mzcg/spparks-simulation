

#example parent folder (demo mimic the real simulation result folder)
# C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\simulation_results_file_exchange_demo
#the above folder contains example folder and files

import os
import shutil

def get_simu_folder_values(folder_name):
    """
    This function to take a folder name (not path) and return the parameter values extracted from the foldername
    Output:
    - speed
    - mpwidth
    - haz
    - thickness

    These output will be used for matching the values to other files (dump file name matching)
    """

    parameter_values = folder_name.split("_")

    #return results in a dictionary format
    return {
        'speed': parameter_values[1],
        'mpwidth': parameter_values[3],
        'haz': parameter_values[5],
        'thickness': parameter_values[7]
    }


#folder_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\simulation_results_file_exchange_demo'
#test_file_path = folder_path+'\speed_3_mpwidth_10_haz_40_thickness_7'
#get_file_name = test_file_path.split("\\")[-1]  #getting only the file name without path
#print(get_file_name)
#print(get_folder_name(get_file_name))

parent_folder_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\simulation_results_file_exchange_demo'  #path to parent folder
log_file_rearrange_error = "file_rearrange_error.log"
log_file_rearrange_error_path = os.path.join(parent_folder_path, log_file_rearrange_error)

log_file_success = "success_file_rearrange_report.log"
log_file_success_path = os.path.join(parent_folder_path, log_file_success)

def main (parent_folder_path, log_file_rearrange_error_path, log_file_success_path):
    with open(log_file_rearrange_error_path, 'a' ) as log_file:
        #iterate through subfolders
        for simu_folder_name in os.listdir(parent_folder_path):  #simu_folder_name eg: speed_23_mpwidth_35_haz_82_thickness_11
            simu_folder_path = os.path.join(parent_folder_path, simu_folder_name)
            print(simu_folder_name)
            if os.path.isdir(simu_folder_path):
                folder_parameter_values = get_simu_folder_values(simu_folder_name)
                for work_file in os.listdir(simu_folder_path):
                    if work_file.endswith(".dump"): #getting only the dump files inside each simulation folder
                        dump_parameter_values = work_file.split("_")
                        print(work_file)  # in here only dump file name displayed: 3D_AM_speed_13_mpWidth_25_haz_91_thickness_5_1.dump
                        if len(dump_parameter_values) >= 8:  #getting parameter values from each dump file.
                            dump_speed = dump_parameter_values[3]
                            dump_mpwidth = dump_parameter_values[5]
                            dump_haz = dump_parameter_values[7]
                            dump_thickness = dump_parameter_values[9]

                            correct_folder_name = f"speed_{dump_speed}_mpwidth_{dump_mpwidth}_haz_{dump_haz}_thickness_{dump_thickness}"  #define folder that current dump file should belong to iun case the current folder miss.
                            correct_folder_path = os.path.join(parent_folder_path, correct_folder_name)

                            #if not os.path.exists(correct_folder_path):
                            #    print(f"Warning: Couldn't find the corresponding folder for {work_file}")
                            #    print(f"Dump file '{work_file}' will remain in its current folder.")
                            #    print(f"Correct folder doesn't exist: '{work_file}'")
                            #    log_file.write(f"Correct folder doesn't exist: '{work_file}'\n")
                            #    continue

                            if (dump_speed != folder_parameter_values['speed'] or
                                dump_mpwidth != folder_parameter_values['mpwidth'] or
                                dump_haz != folder_parameter_values['haz'] or
                                dump_thickness != folder_parameter_values['thickness']):

                                if not os.path.exists(correct_folder_path):
                                #    os.makedirs(correct_folder_path)
                                    #print(f"Warning: Couldn't find the corresponding folder for {work_file}")
                                    #print(f"Dump file '{work_file}' will remain in its current folder.")
                                    print(f"Error! correct folder not found: dump file '{work_file}' remains in current simulation folder '{simu_folder_name}'")
                                    log_file.write(f"Error! correct folder not found: dump file '{work_file}' remains in current simulation folder '{simu_folder_name}'\n")
                                    continue
                                shutil.move(os.path.join(simu_folder_path, work_file), correct_folder_path)
                                with open(log_file_success_path,'a') as log_success: #stored the completed moving (successfully done) to different log file.
                                    log_success.write(f"Done! dump file '{work_file}' moved from wrong folder '{simu_folder_name}' to correct folder '{correct_folder_name}'\n")







main(parent_folder_path, log_file_rearrange_error_path, log_file_success_path)

