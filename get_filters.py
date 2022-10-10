import csv, re, itertools  
from collections import Counter
import pymorphy2

morphy = pymorphy2.MorphAnalyzer(lang='ru')
def get_combs(s:str, l: int):
    global morphy
    for i in itertools.combinations_with_replacement(s, l):
        for j in range(l):
            out = ''.join(i)[:j]
            if not out.isdigit():
                if not morphy.word_is_known(out):
                    continue
            yield out
def get_crossing(str1, strs):
    res = ''
    for s in str1:
        a = True
        for s1 in strs:

            if not s in s1:
                a = False
        if a:
            res += s
        else:
            break
    return res

def word_process(s: str):
    res = morphy.parse(s)[0]
    strs = list(set(map(lambda x: x.word, res.lexeme)))
    return get_crossing(s, strs)


def get_filters(path:str):
    global get_crossing, word_process
    filters = []
    words_lists = []
    text = ''

    words = []
    symbols = []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        lines = list(reader)[1:]
        for i, row in enumerate(lines):
            line = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Zа-яёА-ЯЁ\d]', ' ', row[1])).strip()
            words_lists.append(line)
        text = dict(Counter((' '.join(words_lists)).lower().split(' ')))
        for key, value in text.items():
            if value > 30 and len(key) >= 3:
                words.append(key)
        for n, w in enumerate(words):
            if n > 500:
                break
            r = word_process(w)
            if len(r) >= 3:
                filters.append(r)

        text_2 = dict(Counter(list((re.sub(r'\s+', '', ''.join(words_lists))).lower())))
        for key, value in text_2.items():
            if value > 50 and value < 10000:
                symbols.append(key)
        for n, s in enumerate(symbols):
            if n > 3000:
                break
            filters.append(s)
        t = (' '.join(words_lists)).lower()
        for i in get_combs('0123456789', 5):
            b = t.count(i)
            if b > 30:
                filters.append(i) 

        return sorted(filters)

if __name__ == "__main__":
    print(get_filters())