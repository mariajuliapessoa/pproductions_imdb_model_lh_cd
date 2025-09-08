import os

def generate_structure(path, indent=0):
    result = ""
    for item in sorted(os.listdir(path)):
        item_path = os.path.join(path, item)
        result += "  " * indent + "|-- " + item + "\n"
        if os.path.isdir(item_path):
            result += generate_structure(item_path, indent + 1)
    return result

project_path = r"C:\Users\maria\OneDrive\Área de Trabalho\LH_CD_MARIAJULIAPESSOA"
structure_txt = generate_structure(project_path)

with open("project_structure.txt", "w") as f:
    f.write(structure_txt)

print("Project structure saved to project_structure.txt")
