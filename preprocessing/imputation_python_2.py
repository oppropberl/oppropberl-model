# Merge imputed files in step 1

import glob
import random

single_instance_replacement_keys = [] # on the other hand variable_1 needs to be replaced everywhere in the code

list_of_py_files = []

for f in glob.glob('imputed_files_py/*.py', recursive=True):
    list_of_py_files.append(f)

all_fuzz_tags = []

for _ in range(20):
    for considering_file in list_of_py_files:
        # considering_file = list_of_py_files[0]
        print(considering_file)

        file_name1 = considering_file.split("/")[-1]
        file_name1 = file_name1.split(".")[0]
        status1 = file_name1.split("_")[-1]
        fuzz_tag1 = file_name1.split("_")[0]
        file_name1 = "_".join(file_name1.split("_")[1:-1]) # ignore the error status & fuzz tag
        reading_file = open(considering_file, "r")
        original_file = reading_file.read()
        
        considering_file2 = random.choice(list_of_py_files)
        file_name2 = considering_file2.split("/")[-1]
        file_name2 = file_name2.split(".")[0]
        status2 = file_name2.split("_")[-1]
        fuzz_tag2 = file_name2.split("_")[0]
        file_name2 = "_".join(file_name2.split("_")[1:-1]) # ignore the error status & fuzz tag
        reading_file2 = open(considering_file2, "r")
        original_file2 = reading_file2.read()

        fuzz_tags1 = f"{fuzz_tag1}_{fuzz_tag2}_{file_name1}_{file_name2}"
        fuzz_tags2 = f"{fuzz_tag2}_{fuzz_tag1}_{file_name2}_{file_name1}"
        modified_file = original_file+"\n"+original_file2+"\n" # merge the two files
        
        print(fuzz_tags1, fuzz_tags2)

        if fuzz_tags1 in all_fuzz_tags or fuzz_tags2 in all_fuzz_tags: # either one of the combinations
            print("skipped")
            continue

        all_fuzz_tags.append(fuzz_tags1)
        all_fuzz_tags.append(fuzz_tags2)
        
        reading_file.close()
        reading_file2.close()

        # print(f"{fuzz_tags}_{class_name}", f"{fuzz_tags}_{file_name}")
        
        if "fail" in status1 or "fail" in status2:
            final_status = "fail"
        else:
            final_status = "pass"

        file1 = open(f"imputed_files_py/{fuzz_tags1}_{final_status}.py", "w")
        file1.write(modified_file)
        file1.close()