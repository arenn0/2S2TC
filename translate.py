from googletrans import Translator

file = open("_", "r", encoding='utf8')
file_write = open("_", "w", encoding='utf8')
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
        translated = translator.translate(row, dest='en')
        newrow = translated.text
    except Exception as e:
        print(str(e))
        continue
    translatedList.append(newrow)
    print(newrow)
    file_write.write(newrow + "\n")

