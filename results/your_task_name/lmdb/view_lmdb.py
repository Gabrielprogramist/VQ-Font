import lmdb
import os

def print_lmdb_contents(db_path):
    # Проверка наличия файла lock.mdb
    if not os.path.isfile(os.path.join(db_path, 'data.mdb')):
        print(f"Файл lock.mdb не найден в {db_path}")
        return
    
    # Открытие базы данных в режиме только для чтения
    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        # Итерация по всем ключам и значениям
        cursor = txn.cursor()
        for key, value in cursor:
            print(f"Key: {key}, Value: {value}")

# Укажите путь к вашему файлу .mdb
db_path = 'C:/Users/Timing/Documents/GitHub/VQ-Font/results/your_task_name/lmdb'
print_lmdb_contents(db_path)
