import os
import subprocess

# Create a shell script
with open('script.bat', 'w') as f:
    f.write('pip install numpy')

# Make the script executable
os.chmod('script.bat', 0o755)

# Execute the shell script
subprocess.run(['script.bat'])

#----------------------------------------------------

# Open a shell script
with open('script.bat', 'w') as f:
    f.write('pip install opencv-python')

# Make the script executable
os.chmod('script.bat', 0o755)

# Execute the shell script
subprocess.run(['script.bat'])
