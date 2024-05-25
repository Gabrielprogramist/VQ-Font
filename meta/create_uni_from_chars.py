english_chars_digit = list(range(65, 91)) + list(range(97, 123))   # A-Z, a-z
russian_chars = list(range(0x0410, 0x0450))+[0x0401, 0x0451]  # А-я
kazakh_chars = [0x04D8, 0x04D9, 0x0492, 0x0493, 0x049A, 0x049B, 0x04A2, 0x04A3, 0x04E8, 0x04E9, 0x04B0, 0x04B1, 0x04AE, 0x04AF, 0x04BA, 0x04BB, 0x0406, 0x0456]
########################################################
gen_chars =  [chr(i) for i in english_chars_digit+russian_chars+kazakh_chars]
uni_chars =  [hex(ord(chr(i)))[2:].upper() for i in english_chars_digit+russian_chars+kazakh_chars]
print(uni_chars)


