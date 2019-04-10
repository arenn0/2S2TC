import json
i = 0
weights = []
write = open('italian_output2.txt', 'w', encoding='utf8')
for line in open('italian_output.json', 'r'):
    i += 1
    print('\r {}'.format(i), end='')
    line_json =  json.loads(line)
    for j in range(len(line_json['features'])):
        write.write(line_json['features'][j]['token'] + " " + " ".join([str(k) for k in line_json['features'][j]['layers'][-1]['values']]))
        write.write('\n')

