import glob, subprocess
from tqdm import tqdm

# reading_file = open("/mnt", "r").read()
# print(reading_file)

list_of_java_files = []

for f in glob.glob('imputed_files/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in tqdm(list_of_java_files):
    # considering_file = list_of_java_files[0]
    print(considering_file)

    file_name = considering_file.split("/")[-1]
    class_name = file_name[:-5]
    reading_file = open(considering_file, "r")
    original_file = reading_file.read()

    cmd = f'$CHECKERFRAMEWORK/checker/bin/javac -processor nullness -Acfgviz=org.checkerframework.dataflow.cfg.visualize.StringCFGVisualizer "{considering_file}" > imputed_files_cfg/{class_name}.txt'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # if len(stdout.decode().strip()) != 0 or len(stderr.decode().strip()) != 0:
    #     print("Error!")
    #     print(stdout.decode().strip())
    #     print(stderr.decode().strip())
    #     break
    