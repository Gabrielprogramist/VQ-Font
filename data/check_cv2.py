from PIL import Image
import numpy as np
import os
import cv2

# Пример использования абсолютного пути
current_directory = os.getcwd()
# Убедитесь, что путь к файлу в Unicode
image_path = os.path.join(current_directory, 'data/style/train/vGothn/lower_ё_vGothn.png')

# Нормализация пути
image_path_normalized = os.path.normpath(image_path)

try:
    # Открытие изображения с помощью PIL
    pil_image = Image.open(image_path_normalized)
    image = np.array(pil_image)

    # Преобразование в формат, совместимый с OpenCV (если необходимо)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    print(f"Successfully read image: {image_path_normalized}")
except Exception as e:
    print(f"Failed to read image: {image_path_normalized}. Error: {e}")
