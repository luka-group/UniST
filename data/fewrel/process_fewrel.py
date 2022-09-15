import json

rid2label = {}
with open('pid2name.json', 'r') as f:
    d = json.loads(f.read().splitlines()[0])
    for rid, (label, _) in d.items():
        rid2label[rid] = label

for name in ['train_wiki', 'val_wiki']:
    with open(name+'.json', 'r', encoding='utf-8') as f:
        d = json.loads(f.read().splitlines()[0])
        
    output = []
    for rid in d:
        for data in d[rid]:
            tokens = data['tokens']    
            head_span = [data['h'][2][0][0], data['h'][2][0][-1]+1]
            tail_span = [data['t'][2][0][0], data['t'][2][0][-1]+1]
            label = rid2label[rid]
            output.append({'tokens':tokens, 'head_span':head_span, 'tail_span':tail_span, 'label':label})
            
    with open(name+'_processed.jsonl', 'w', encoding='utf-8') as f:
        for line in output:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
