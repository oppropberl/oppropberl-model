# Remove new line char after annot from the annotated files + remove double annotation (because of existing annotation) -> delete the second one + removes all indentation

import glob, subprocess
from platform import java_ver
from tqdm import tqdm

list_of_java_files = []

for f in glob.glob('./annotated/*.java', recursive=True):
    list_of_java_files.append(f)

for considering_file in list_of_java_files:
    file_name = considering_file.split("/")[-1]
    reading_file = open(considering_file, "r")
    new_file_content = "import org.checkerframework.checker.nullness.qual.*;\n"

    for line in reading_file:
        new_line = line.strip()
        if new_line in ["@NonNull", "@Nullable"]:
            print("Found: ", new_line)
            #new_line = new_line.replace("@NonNull", "")
            new_line += " "
            new_file_content += new_line

        elif "import nninf.qual" in new_line: 
            new_line = new_line.replace("import nninf.", "import org.checkerframework.checker.nullness.")
            new_line += "\n"
            new_file_content += new_line
            
        elif ":: warning:" in new_line or ":: error:" in new_line:
            pass
            
        else:
            new_line += "\n"
            new_file_content += new_line

    reading_file.close()

    new_file_content = new_file_content.replace("@Nullable @NonNull", "@Nullable")
    new_file_content = new_file_content.replace("@NonNull @Nullable", "@NonNull")
    new_file_content = new_file_content.replace("@Nullable @Nullable", "@Nullable")
    new_file_content = new_file_content.replace("@NonNull @NonNull", "@NonNull")

    file1 = open(f"/home/anonymous/DLAnnot/annotated_clean/{file_name}", "w")
    file1.write(new_file_content)
    file1.close()