import lmdb
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def print_lmdb_contents(db_path):
    # Проверка наличия файла data.mdb
    if not os.path.isfile(os.path.join(db_path, 'data.mdb')):
        print(f"Файл data.mdb не найден в {db_path}")
        return
    
    # Открытие базы данных в режиме только для чтения
    env = lmdb.open(db_path, readonly=True)
    with env.begin() as txn:
        # Итерация по всем ключам и значениям
        cursor = txn.cursor()
        for key, value in cursor:
            print(f"Key: {key}")
            # Декодирование изображения из value
            image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                # Отображение изображения
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # plt.title(f"Image for key: {key.decode('utf-8')}")
                # plt.show()
            else:
                print(f"Failed to decode image for key: {key}")

# Укажите путь к вашему файлу .mdb
db_path = 'C:/Users/Timing/Documents/GitHub/VQ-Font/results/your_task_name/lmdb'
print_lmdb_contents(db_path)
