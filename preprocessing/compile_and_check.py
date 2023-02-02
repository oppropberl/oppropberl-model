import glob
import subprocess

list_of_java_files_n = [] # no errors
list_of_java_files_y = [] # errors

# ignore_flag = "/mnt/c/Users/anonymous/OneDrive - anonymous/Research Related/anonymous/Annotation Prediction using DL/Raw files/Issue3275/NullCheck15_y.java"

for f in glob.glob('/mnt/c/Users/anonymous/OneDrive - anonymous/Research Related/anonymous/Annotation Prediction using DL/**/*.java', recursive=True):
    if f[-7:-5] == "_n":
        list_of_java_files_n.append(f)
    else:
        list_of_java_files_y.append(f)

# print(list_of_java_files_y[:1])

index_start = 0 #list_of_java_files_y.index(ignore_flag)

for file in list_of_java_files_n[index_start:]:   
    cmd = f'$CHECKERFRAMEWORK/checker/bin/javac -processor nullness "{file}"'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    print(file)
    if len(stderr.decode().strip()) != 0:
        print("out: ", stderr.decode().strip())
        exit(1)

for file in list_of_java_files_y[index_start:]:
    cmd = f'$CHECKERFRAMEWORK/checker/bin/javac -processor nullness "{file}"'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print("out: ", stderr.decode().strip())

    print(file)
    if len(stderr.decode().strip()) == 0:
        print("out: ", stderr.decode().strip())
        exit(1)

    reading_file = open(file, "r")
    error_count = 0
    error_lines = []
    for line_no, line in enumerate(reading_file):
        new_line = line.strip()
        if ":: error:" in new_line:
            error_lines.append(line_no+2) # 0 offset & error message is at a line prior 
            error_count+=1
    
    # print(error_count, error_lines)
    error_count_cf = int(stderr.decode().strip().split("\n")[-1].split(" ")[0])

    if error_count_cf != error_count:
        print(error_count, error_count_cf)
        print("out: ", stderr.decode().strip())
        exit(1)
    