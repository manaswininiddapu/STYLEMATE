import subprocess

# Path to the folder you want to open in VSCode
folder_path = "E:\2D_Virtual_try-on\results"

# Construct the command to open the folder in VSCode
command = ["code", folder_path]

# Use subprocess to run the command
subprocess.run(command, check=True)