#ref: https://github.com/kvmani/machineLearning/blob/35a0537554ffb4180d146c390f85df8ddac776e5/src/grainStructure/splittingData.py
#note (June 1 2024): after generation, we move the files in test folder to train, and copy the val files to test file. basically, test and val using the same data. (based on current settting of model)
import os
import random
import shutil


#data_parent_folder= r"E:\Data\data_augmentation_all\data_augmentation_xy_0_32_64" #xy_0_32_64 dataset
data_parent_folder= r"D:\Zhaochen\ML_training_data_without_augmentation\Procssed_original_xy_xz_64" #xy_xz_64 dataset

#dataset_folder_name = "merged_data_xy_0_32_46"  # case1: all 5 effects, total images 5 * 4704 = 23520 images
#dataset_folder_name = "merged_data_3effects_xy_0_32_46"  # case 2:  3-effects, total images 3* 4704 = 14112 images
dataset_folder_name = "stitchedImages"  # test case: just want original data without any effects

source_folder = os.path.join(data_parent_folder, dataset_folder_name) # Source folder path
#destination_folder = os.path.join(data_parent_folder, "splittedData")  # case 1: Destination folder path (for all 5 effects)
destination_folder = os.path.join(data_parent_folder, "splittedData_original_xy_xz_64")  # case 2: Destination folder path (for only 3 effects)

# Create output folder based on the provided destination_folder path
os.makedirs(destination_folder, exist_ok=True) #create the splittedData folder
# Create output subfolders to save the splitted data later
subfolders = ["train",'val','test']
for subfolder in subfolders:
    subfolder_path = os.path.join(destination_folder, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
print(f"Created {destination_folder} with subfolders: {', '.join(subfolders)}")



def calculate_split(input_value, train_ratio, val_ratio, test_ratio):
    # Calculate initial values
    train_count = int(input_value * train_ratio)
    val_count = int(input_value * val_ratio)
    test_count = int(input_value * test_ratio)

    # Ensure that the sum of train, val, and test counts equals the input value
    total_assigned = train_count + val_count + test_count
    remaining_samples = input_value - total_assigned

    # Distribute remaining samples to the training set to ensure all images are used
    train_count += remaining_samples

    return train_count, val_count, test_count

def count_total_data_size(folder_path):
    """
    This function aim to calculate how many image data inside a pre-split folder (total amount of images we used for whole ML modeling
    """
    file_count = sum(1 for file in os.listdir(folder_path))
    return int(file_count)

# Function to move files randomly from source to destination folder
def move_files(source, destination, count):
    files = os.listdir(source)
    random.shuffle(files)
    selected_files = files[:count]
    for file_name in selected_files:
        source_path = os.path.join(source, file_name)
        destination_path = os.path.join(destination, file_name)
        shutil.move(source_path, destination_path)
        print(f"Moved {file_name} to {destination}")

def move_files_keep_val_test_set_same(train_folder, test_folder, val_folder):
    try:
        # Move files from val to train folder
        files_to_move = os.listdir(val_folder)
        for file_name in files_to_move:
            src = os.path.join(val_folder, file_name)
            dst = os.path.join(train_folder, file_name)
            shutil.move(src, dst)
        print(f"Moved {len(files_to_move)} files from {val_folder} to {train_folder}")

        # Copy files from test to val folder
        files_to_copy = os.listdir(test_folder)
        for file_name in files_to_copy:
            src = os.path.join(test_folder, file_name)
            dst = os.path.join(val_folder, file_name)
            shutil.copy(src, dst)
        print(f"Copied {len(files_to_copy)} files from {test_folder} to {val_folder}")

    except Exception as e:
        print(f"An error occurred: {e}")



def main():
    """
    Part A: Getting input and output path (Already done in the begining of this program)
       step 1: get source folder path (total data set before split)
       step 2: create destination folder path (to save splitted data after split)
       -- step 2a: create the destination folder based on the path
       -- step 2b: create subfolders within the destination folder (train, val, test)
    Part B: Getting split data values
       step 3: iterate the files in source folder to get total number of data (function: count_total_data_size())
       step 4: calculate the number of data for train, val, and test respectively based on the given ratio (function: calculate_split())
       step 5: use the calculated number put into setting (file_counts dictionary)
    Part C: move files to destination folder
       step 6: shuffle files and move files to corresponding folder (train,val,test) (function: move_files())
    ** Part D (Optional):
       ** Due to current model setting, we want to move files from "test" to "train", and copy the files from "val" to "test" (val, test using same data)
       ** You can also do it manually if you want
       step 7: moves files from "test" to "train", and copy files from "val" to "test" (function: move_files_keep_val_test_same())

    """

    # Part A: done in the beginning of the program

    # Part B: getting split data values
    total_data_size = count_total_data_size(source_folder)  # count total number of files inside the source data folder
    train_num, test_num, val_num = calculate_split(total_data_size, train_ratio=-.8, val_ratio=0.1,test_ratio=0.1)  # caculate how many data for each sets
    print("train:", train_num, "test:", test_num, "val:", val_num) # eg: for 23250 data, train: 18816 test: 2352 val: 2352
    # Dictionary specifying the number of files for train, test, and validation sets
    file_counts = {'train': train_num, 'test': test_num, 'val': val_num}  # set the int number of each dataset

    # Part C: Move files to destination folder
    for split, count in file_counts.items():  # Move files based on specified counts
        move_files(source_folder, os.path.join(destination_folder, split), count)

    # Optional: Part D
    train_folder_path = os.path.join(destination_folder, "train")
    val_folder_path = os.path.join(destination_folder, "val")
    test_folder_path = os.path.join(destination_folder,'test')

    move_files_keep_val_test_set_same(train_folder_path, test_folder_path, val_folder_path)
    return 0

if __name__ == "__main__":
    main()