# import subprocess
#
# command = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\command_line_arg.sh -speed 3 -mpwidth 69 -haz 114.0 -thickness 10.0"
# command2 = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\command_line_arg.sh -speed 5 -mpwidth 7 -haz 11.0 -thickness 13.0"
# subprocess.run(command, check=True, shell=True)
# subprocess.run(command2, check=True, shell=True)
#
# print(type(command))

import os

# Using a raw string
path1 = r"C:\Users\zg0017\PycharmProjects\spparks-simulation\examples\am_path\pattern_repeat\generated_SPPARKS_scripts"

# Joining paths manually with forward slashes
path2 = "/mnt/c/Users\\zg0017\\PycharmProjects\\spparks-simulation\\examples\\am_path\\pattern_repeat\\spk_mpi"

# Using os.path.join to maintain consistency
path3 = os.path.join("/mnt/c", "Users", "zg0017", "PycharmProjects", "spparks-simulation", "examples", "am_path", "pattern_repeat", "spk_mpi")

print("Path 1:", path1)
print("Path 2:", path2)
print("Path 3:", path3)
