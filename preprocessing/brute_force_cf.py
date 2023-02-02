# mask all annotated_clean files

import glob, subprocess
from platform import java_ver
from tqdm import tqdm
import re

list_of_java_files = []

for f in glob.glob('./annotated_mask/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in tqdm(list_of_java_files):
    class_name = considering_file.split("/")[-1][:-5]
    original_file = open(considering_file, "r").read()

    mask_count = original_file.count('<mask>')
    print(class_name, mask_count)
    mask_replacements = ["@Nullable", "@NonNull"]

    for sub_file_number in range(2**mask_count): # all possible combinations in the mask
        binary_eq = list(bin(sub_file_number)[2:]) # 10 -> 1, 0, 1, 0 -> combination for NonNull/Nullable
        binary_eq = ['0']*(mask_count - len(binary_eq)) + binary_eq # padding zeros to make all of equal length
        reading_file = original_file
        for val in binary_eq:
            reading_file = reading_file.replace("<mask>", mask_replacements[int(val)], 1)

        reading_file = reading_file.replace(class_name, f"{class_name}_{''.join(binary_eq)}")

        file1 = open(f"/home/anonymous/DLAnnot/temp/{class_name}_{''.join(binary_eq)}.java", "w")
        file1.write(reading_file)
        file1.close()

        cmd = f'$CHECKERFRAMEWORK/checker/bin/javac -processor nullness ./temp/{class_name}_{"".join(binary_eq)}.java'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        print(stdout)
        print(stderr.decode().strip())

        if "error:" in stderr.decode().strip(): # error
            error_tag = "cffail"
            unique_err = {}
            match = re.findall(r"(\w+).java:(\d+): error: (\S+)+", stderr.decode().strip())

            for match_item in match:
                unique_err[int(match_item[1])] = match_item[2].replace("[", "(").replace("]", ")")

            keys = list(unique_err.keys())
            keys.sort(reverse=True) # going in reverse order to not mess with the line numbers after adding the error tag

            lines = reading_file.split("\n")
            for key in keys:
                lines.insert(key-1, f"// :: error: {unique_err[key]}")
            reading_file = "\n".join(lines)

        else:
            error_tag = "cfpass"

        print(f"Done: {class_name}_{error_tag}_{sub_file_number} @ {binary_eq}")

        reading_file = reading_file.replace(f"{class_name}_{''.join(binary_eq)}", f"{class_name}_{error_tag}_{sub_file_number}")

        file1 = open(f"/home/anonymous/DLAnnot/annotated_brute_force_cf/{class_name}_{error_tag}_{sub_file_number}.java", "w")
        file1.write(reading_file)
        file1.close()
