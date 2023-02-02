import glob, subprocess
from platform import java_ver
from tqdm import tqdm

list_of_java_files1, list_of_java_files2 = [], []


for f in glob.glob('/mnt/p/RW/annotation-pred-data/raw_files/**/*.java', recursive=True):
    list_of_java_files1.append(f.split("/")[-1])

for f in glob.glob('./annotated/*.java', recursive=True):
    list_of_java_files2.append(f.split("/")[-1])

print(len(list_of_java_files1), len(list_of_java_files2))

for i in list_of_java_files1:
    if i not in list_of_java_files2:
        print(i)

print()

for i in list_of_java_files2:
    if i not in list_of_java_files1:
        print(i)