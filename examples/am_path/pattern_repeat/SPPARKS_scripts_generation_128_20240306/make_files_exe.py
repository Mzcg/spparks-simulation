import os
import subprocess

# Get the current working directory
current_folder = os.getcwd()

# List all files in the current folder
all_files = os.listdir(current_folder)

# Filter files that end with ".sh"
sh_files = [file for file in all_files if file.endswith(".sh")]

# Make each .sh file executable
for sh_file in sh_files:
    file_path = os.path.join(current_folder, sh_file)
    subprocess.run(["chmod", "+x", file_path])
    print(f"Made {sh_file} executable.")