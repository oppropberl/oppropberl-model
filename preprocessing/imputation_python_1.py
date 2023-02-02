import glob
import random

fuzz_dict = {
    "df": ("A", ["tab1", "df1", "var_df"]),
    ".head()": ("B", [".tail()", ".to_csv('out.csv')"]),
    "var1": ("C", ["var_1", "var_a", "data_w", "value_m"]),
    "# dummy_lines": ("J", ["\n", "\nvar_0g = -1 \nif var_0g < 0: var_ex = 0", "\nvar_0h = True \nif var_0h != 0: var_ex = None", "var_0i = 'this is a string'"]),
    "print(": ("D", ["display("]),
} 

single_instance_replacement_keys = [] # on the other hand variable_1 needs to be replaced everywhere in the code

list_of_py_files = []

for f in glob.glob('python_error_pred/*.py', recursive=True):
    list_of_py_files.append(f)

for considering_file in list_of_py_files:
    # considering_file = list_of_py_files[0]
    print(considering_file)

    file_name = considering_file.split("/")[-1]
    reading_file = open(considering_file, "r")
    original_file = reading_file.read()

    all_fuzz_tags = []

    for _ in range(10): 
        fuzz_tags = ""
        modified_file = original_file

        for key in fuzz_dict.keys():
            while key in modified_file:
                list_of_choices = [None]+fuzz_dict[key][1]
                to_replace = random.choice(list_of_choices)
                fuzz_tags += f"{fuzz_dict[key][0]}{list_of_choices.index(to_replace)}"
                # print(key, to_replace, fuzz_tags)

                if to_replace is None:
                    break # keep original

                if key in single_instance_replacement_keys:
                    modified_file = modified_file.replace(key, to_replace, 1)
                else:
                    modified_file = modified_file.replace(key, to_replace) # replace everywhere
                    break 

        # print(modified_file)
        # print(fuzz_tags)

        if fuzz_tags in all_fuzz_tags:
            print(fuzz_tags, " skipped")
            continue

        all_fuzz_tags.append(fuzz_tags)

        reading_file.close()

        # print(f"{fuzz_tags}_{class_name}", f"{fuzz_tags}_{file_name}")

        file1 = open(f"imputed_files_py/{fuzz_tags}_{file_name}", "w")
        file1.write(modified_file)
        file1.close()