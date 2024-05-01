#############################
#Objective: the main goal for this program is to extract last dump file

# goal 1: just get the last dump file name and print the list (maybe output to a txt file)
import os

folder_path = r'C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\clean_simulation_results_demo'  #path to parent folder  #demo

def get_last_dump_file(parent_folder_path):
    dump_file_list = []
    for item in os.listdir(parent_folder_path): #iterate all the files inside the folder
        itempath = os.path.join(parent_folder_path, item)

        #check if the filepath is a directory, if yes, continue operations;
        if os.path.isdir(itempath):
            max_index = -1 #initialize the index to -1
            last_dump_file_path = None # initialize to now.

            for simu_files in os.listdir(itempath): #filtered the simulation folder, simu_files are any files within the simu folder. not just dump file.
                if simu_files.endswith(".dump"): #only consider .dump files.
                    index_part  = simu_files.split("_")[-1] # 1.dump
                    index_val = int(index_part.split(".")[0]) #get only the dump file index value

                    #update the index until find the max one
                    if index_val > max_index:
                        max_index = index_val
                        last_dump_file_name = simu_files
                        last_dump_file_path = os.path.join(itempath, last_dump_file_name) #e.g: C:\Users\zg0017\PycharmProjects\spparks-simulation\tools\clean_simulation_results_demo\speed_13_mpwidth_25_haz_56_thickness_5\3D_AM_speed_13_mpWidth_25_haz_56_thickness_5_9.dump
            if last_dump_file_path: #only append if it's not None (avoid the first element turn to be None in the list)
                dump_file_list.append(last_dump_file_path) #only store the last dump file to the list, not all files.
            #print(last_dump_file_path) #for result display (path for each last dump file)

    #print(dump_file_list) #the backslash will doubled. but when you access single item, it will get back to #for display: a list of last dump file path
    return dump_file_list


#main function call (i am just lazy to write main function)
get_last_dump_file(folder_path)
