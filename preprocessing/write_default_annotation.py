import glob, subprocess
from tqdm import tqdm

# reading_file = open("/mnt", "r").read()
# print(reading_file)

list_of_java_files = []
last_file_before_err = None #"/mnt/p/RW/annotation-pred-data/raw_files/Loops/Loops20_y.java"
found_file = True #False

for f in glob.glob('/mnt/p/RW/annotation-pred-data/raw_files/**/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in tqdm(list_of_java_files):
    # considering_file = list_of_java_files[0]
    print(considering_file)
    if not found_file and considering_file != last_file_before_err:
        continue
    else:
        found_file = True

    file_name = considering_file.split("/")[-1]
    class_name = file_name[:-5]
    reading_file = open(considering_file, "r")
    original_file = reading_file.read()

    cmd = f'$CFI/scripts/inference-dev --checker nninf.NninfChecker --solver checkers.inference.solver.MaxSat2TypeSolver --solverArgs="collectStatistics=true,outputCNF=true" --hacks=true --writeDefaultAnnotations=true -m ROUNDTRIP -afud /home/anonymous/DLAnnot/annotated {considering_file}'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if "Insert annotations succeeded" not in stdout.decode().strip():
        print("ERROR!")
        print(cmd)
        break
