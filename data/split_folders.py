import os
import shutil
import random

def split_folders(input_directory, train_directory, val_directory, train_count=5562, val_count=1390):
    # Проверяем, существует ли выходная директория, если нет - создаем их
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)
    if not os.path.exists(val_directory):
        os.makedirs(val_directory)

    # Получаем список всех папок в исходной директории
    all_folders = [folder for folder in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, folder))]
    
    # Проверяем, что количество папок соответствует ожиданиям
    assert len(all_folders) == train_count + val_count, "Общее количество папок не соответствует ожиданиям"

    # Случайным образом перемешиваем список папок
    random.shuffle(all_folders)

    # Разделяем папки на train и val
    train_folders = all_folders[:train_count]
    val_folders = all_folders[train_count:train_count + val_count]

    # Перемещаем папки в соответствующие директории
    for folder in train_folders:
        src_path = os.path.join(input_directory, folder)
        dst_path = os.path.join(train_directory, folder)
        shutil.move(src_path, dst_path)
        print(f"Перемещена папка {src_path} в {dst_path}")

    for folder in val_folders:
        src_path = os.path.join(input_directory, folder)
        dst_path = os.path.join(val_directory, folder)
        shutil.move(src_path, dst_path)
        print(f"Перемещена папка {src_path} в {dst_path}")

# Пример использования
input_directory = 'data/style'  # Замените на путь к вашей исходной директории
train_directory = 'data/style2/train'  # Замените на путь к директории для train
val_directory = 'data/style2/val'      # Замените на путь к директории для val

split_folders(input_directory, train_directory, val_directory)
