import glob
import random

"""
variable_object
variable_1 2 3 4 5
method_1
private/protected
//dummy_if_check -> else if false line can be added
//dummy_else_check
//dummy_lines
class_1
classname
toString??

"""

fuzz_dict = {
    "variable_object": ("A", ["obj1", "object1", "obj"]),
    "variable_1": ("B", ["z1", "var_a", "data_q", "value_l"]),
    "variable_2": ("C", ["z2", "var_b", "data_w", "value_m"]),
    "variable_3": ("D", ["z3", "var_c", "data_e", "value_n"]),
    "variable_4": ("E", ["z4", "var_d", "data_r", "value_o"]),
    "variable_5": ("F", ["z5", "var_e", "data_t", "value_p"]),
    "method_1": ("G", ["m1", "func_1", "check_null", "nullness_test"]),
    "//dummy_if_check": ("H", ["\n", "\n String var_0a = null; \n if (var_0a == null) { String var_ex = null; }", "\n String var_0b = null; \n if (var_0b != null) { String var_ex = null; }", "String var_0c = null;"]),
    "//dummy_else_check": ("I", ["\n", "\n String var_0d = null; \n if (var_0d == null) { String var_ex = null; }", "\n String var_0e = null; \n if (var_0e != null) { String var_ex = null; }", "String var_0f = null;"]),
    "//dummy_lines": ("J", ["\n", "\n String var_0g = null; \n if (var_0g == null) { String var_ex = null; }", "\n String var_0h = null; \n if (var_0h != null) { String var_ex = null; }", "String var_0i = null;"]), # method parameter, boolean vars
    "class_1": ("K", ["C1", "Class1", "CheckNull", "TestNullnessClass"])
} 

single_instance_replacement_keys = ["public", "//dummy_if_check", "//dummy_else_check", "//dummy_lines"] # on the other hand variable_1 needs to be replaced everywhere in the code

list_of_java_files = []

for f in glob.glob('annotated_brute_force_cf/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in list_of_java_files:
    # considering_file = list_of_java_files[0]
    print(considering_file)

    file_name = considering_file.split("/")[-1]
    class_name = file_name[:-5]
    reading_file = open(considering_file, "r")
    original_file = reading_file.read()

    all_fuzz_tags = []

    for _ in range(20): # 35 fuzzed versions for every file
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

        modified_file = modified_file.replace(class_name, f"{fuzz_tags}_{class_name}")

        # print(f"{fuzz_tags}_{class_name}", f"{fuzz_tags}_{file_name}")

        file1 = open(f"imputed_files/{fuzz_tags}_{file_name}", "w")
        file1.write(modified_file)
        file1.close()