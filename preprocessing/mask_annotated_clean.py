# mask all annotated_clean files

import glob, subprocess
from platform import java_ver
from tqdm import tqdm

list_of_java_files = []

for f in glob.glob('./annotated_clean/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in list_of_java_files:
    file_name = considering_file.split("/")[-1]
    reading_file = open(considering_file, "r").read()
    
    reading_file = reading_file.replace("@Nullable", "<mask>")
    reading_file = reading_file.replace("@NonNull", "<mask>")

    file1 = open(f"/home/anonymous/DLAnnot/annotated_mask/{file_name}", "w")
    file1.write(reading_file)
    file1.close()