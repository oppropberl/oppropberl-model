import glob
import subprocess
import pandas as pd

all_files = {}

list_of_java_files_n = [] # no errors
list_of_java_files_y = [] # errors

for f in glob.glob('annotated_brute_force_cf/*.java', recursive=True):
    file_name = f.split("/")[-1]
    file = file_name.split("_")[0]
    if file not in all_files:
        all_files[file] = {}
        all_files[file]['cffail'] = 0
        all_files[file]['cfpass'] = 0
    error = file_name.split("_")[-2]
    if error == 'cffail':
        all_files[file]['cffail'] += 1
    else:
        all_files[file]['cfpass'] += 1

pd.DataFrame.from_dict(all_files, orient='index').to_csv('annotated_brute_force_cf_stats.csv')