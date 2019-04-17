from googletrans import Translator
import copy


file = open("CrisisLexT26_output.txt", "r", encoding='utf8')
file_write = open("CrisisLexT26_english_to_italian_output.txt", "w", encoding='utf8')
# print("Hello")
lines = []
index = 0
for line in file:
    lines.append(line.rstrip())

translatedList = []
for row in lines:
    # REINITIALIZE THE API
    translator = Translator(service_urls=["translate.google.com", 'translate.google.co.kr'])
    newrow = ""
    try:
        # translate the 'text' column
        translated = translator.translate(row, dest='it')
        newrow = translated.text
    except Exception as e:
        print(str(e))
        continue
    translatedList.append(newrow)
    print(newrow)
    file_write.write(newrow + "\n")

