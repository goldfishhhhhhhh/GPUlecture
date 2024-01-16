import os

# Set the directory you want to start from
rootDir = r'E:\Ali\features24'  # This sets the current directory

for i in range(1, 81):
    dirName = os.path.join(rootDir,"video"+str(i))
    try:
        # Create target Directory
        os.mkdir(dirName)
        print(f"Directory {dirName} created.")
    except FileExistsError:
        print(f"Directory {dirName} already exists.")