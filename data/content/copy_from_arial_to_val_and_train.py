import os
import shutil

def uni_to_gen(uni_chars):
    gen_chars = [chr(int(uni, 16)) for uni in uni_chars]
    return gen_chars

# Пример использования
uni_chars = ["5A", "4C", "65", "432", "441", "59", "4A3", "430", "7A", "4B0", "425", "6B", "419", "416", "42B", "75", "444", "63", "43B", "423", "4E8", "401", "55", "42D", "445", "46", "44C", "4AE", "422", "4F", "49A", "47", "420", "76", "410", "4A", "429", "424", "44E", "6F", "50", 
"456", "61", "4A2", "70", "440", "69", "436", "66", "42F", "428", "71", "42A", "41", "79", "43", "442", "74", "44", "45", "4E", "4D9", "443", "43E", "4D", "418", "6A", "49B", "72", "435", "4B1", "415", "41D", "44D", "438", "446", "44A", "41C", "44F", "447", "431", "434", "451", "77", "68", "78", "52", "41E", "41B", "493", "4D8", "439", "427", "48", "413", "58", "53", "43A", "448", "42C", "67", "43C", "56", "62", "41F", 
"492", "4B", "42", "6D", "41A", "54", "449", "49", "44B", "43D"]
gen_chars = uni_to_gen(uni_chars)

def copy_files(file_list, source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    for file_name in file_list:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            print(f"Copied {file_name} to {target_dir}")
        else:
            print(f"File {file_name} not found in {source_dir}")

# Пример использования
file_list = ["lower_" + i + "_arial.png" if i.islower() else "upper_" + i + "_arial.png" for i in gen_chars]
source_dir = 'arial_136'
target_dir = 'arial_115'

copy_files(file_list, source_dir, target_dir)

