import subprocess
import os
import argparse

# Function that used ImageMagick to convert 32 bit integer to 8 bit integer

def convert32bitTo8bit(inputFile, outputFile):
    subprocess.call(['magick', inputFile, '-colors', '255', 'png8:'+outputFile])

# Traverse the directory and convert all the files

def traverseDirectory(directory):
    # Make a "os.path.join(root, "converted")" directory if it doesn't exist
    if not os.path.exists(os.path.join(directory, "converted")):
        os.makedirs(os.path.join(directory, "converted"))
    for root, dirs, files in os.walk(directory):
        for file in files:
            print("Checking file: " + os.path.join(root, "converted", file[:-4]+".png"))
            if file.endswith(".png"):
                convert32bitTo8bit(os.path.join(root, file), os.path.join(root, "converted", file[:-4]+".png"))

# Use argparse to get the directory

parser = argparse.ArgumentParser(description='Convert 32 bit integer to 8 bit integer')
parser.add_argument('-d', '--directory', help='Directory to traverse', required=True)
args = parser.parse_args()

traverseDirectory(args.directory)