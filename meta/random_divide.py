import random

# Исходный список размером 136
original_list = ["41", "42", "43", "44", "45", "46", "47", "48", "49", "4A", "4B", "4C", "4D", "4E", "4F", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5A", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6A", "6B", "6C", "6D", "6E", "6F", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7A", "410", "411", "412", "413", "414", "415", "416", "417", "418", "419", "41A", "41B", "41C", "41D", "41E", "41F", "420", "421", "422", "423", "424", "425", "426", "427", "428", "429", "42A", "42B", "42C", "42D", "42E", "42F", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "43A", "43B", "43C", "43D", "43E", "43F", "440", "441", "442", "443", "444", "445", "446", "447", 
"448", "449", "44A", "44B", "44C", "44D", "44E", "44F", "401", "451", "4D8", "4D9", "492", "493", "49A", "49B", "4A2", "4A3", "4E8", "4E9", "4B0", "4B1", "4AE", "4AF", "4BA", "4BB", "406", "456"]

# Перемешиваем исходный список
random.shuffle(original_list)

# Разделяем на два списка
list_115 = original_list[:115]
list_21 = original_list[115:]

# Проверяем размеры списков
print(f"List 115 size: {len(list_115)}")
print(f"List 21 size: {len(list_21)}")

# Проверяем содержимое списков
print("List 115:", list_115)
print("List 21:", list_21)
