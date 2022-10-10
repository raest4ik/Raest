import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from get_filters import get_filters
import warnings
import re, datetime, csv
import itertools
warnings.filterwarnings("ignore")
filters_1 = get_filters('train.csv')
filters_2 = get_filters('data.csv')
filters = []
for filt in filters_1:
    if filt in filters_2:
        filters.append(filt)
filters = list(set(filters))
def load(path: str, drop_fields: list):
    global filters
    df_train = pd.read_csv(path)
    df_train['description'] = df_train.description.map(lambda x: x.lower())
    for i, word in enumerate(filters):
        df_train[f"param{i}-0"] = df_train.description.map(lambda x: int(word in x))
    df_train["len_description"] = df_train.description.map(len)
    df_train["object"] = df_train["description"].map(lambda x : x.split()[0]) 
    df_train["object"] = pd.Categorical(df_train["object"])
    df_train["object"].astype('category').cat.codes
    df_train["object"] = df_train["object"].cat.codes
    df_train['have_pokm'] = df_train.description.map(lambda x: int('покм' in x))
    df_train['фотог'] = df_train.description.map(lambda x: int('фото' in x))
    df_train['пки'] = df_train.description.map(lambda x: int('пкм' in x))
    df_train['моно'] = df_train.description.map(lambda x: int('монета' in x))
    df_train['картина'] = df_train.description.map(lambda x: int('картина' in x))
    df_train['форм'] = df_train.description.map(lambda x: int('форм' in x))
    df_train['кар'] = df_train.description.map(lambda x: int('кар' in x))
    df_train['член'] = df_train.description.map(lambda x: int('член' in x))
    df_train['стекл'] = df_train.description.map(lambda x: int('стекл' in x))
    df_train['мат'] = df_train.description.map(lambda x: int('мат' in x))
    df_train['цвет'] = df_train.description.map(lambda x: int('цвет' in x))
    df_train['ряд'] = df_train.description.map(lambda x: int('ряд' in x))
    df_train['изо'] = df_train.description.map(lambda x: int('изображен' in x))
    df_train['сни'] = df_train.description.map(lambda x: int('сни' in x))
    df_train['чел'] = df_train.description.map(lambda x: int('человек' in x))
    df_train['пейзаж'] = df_train.description.map(lambda x: int('пейзаж' in x))
    df_train['ь'] = df_train.description.map(lambda x: int('ь' in x))
    df_train['лист'] = df_train.description.map(lambda x: int('лист' in x))
    df_train['глуб'] = df_train.description.map(lambda x: int('19' in x))
    df_train['пер'] = df_train.description.map(lambda x: int('пер' in x))
    df_train['фраг'] = df_train.description.map(lambda x: int('фрагмент' in x))
    df_train['чер'] = df_train.description.map(lambda x: int('чернила' in x))
    df_train['плен'] = df_train.description.map(lambda x: int('плен' in x))
    df_train['дерев'] = df_train.description.map(lambda x: int('дерев' in x))
    df_train['сталь'] = df_train.description.map(lambda x: int('сталь' in x))
    df_train['голов'] = df_train.description.map(lambda x: int('голов' in x))
    df_train['тон'] = df_train.description.map(lambda x: int('тон' in x))
    df_train['объем'] = df_train.description.map(lambda x: int('объем' in x))
    df_train['эмал'] = df_train.description.map(lambda x: int('эмал' in x))
    df_train['крас'] = df_train.description.map(lambda x: int('крас' in x))
    df_train['раст'] = df_train.description.map(lambda x: int('раст' in x))
    df_train['зелен'] = df_train.description.map(lambda x: int('зелен' in x))
    df_train['граф'] = df_train.description.map(lambda x: int('граф' in x))
    df_train['пис'] = df_train.description.map(lambda x: int('пис' in x))
    df_train['корон'] = df_train.description.map(lambda x: int('корон' in x))
    df_train['глин'] = df_train.description.map(lambda x: int('глин' in x))
    df_train['бронз'] = df_train.description.map(lambda x: int('бронз' in x))
    df_train['сереб'] = df_train.description.map(lambda x: int('сереб' in x))
    df_train['федор'] = df_train.description.map(lambda x: int('федор' in x))
    df_train['алек'] = df_train.description.map(lambda x: int('александр' in x))
    df_train['евгений'] = df_train.description.map(lambda x: int('евгений' in x))
    df_train['андр'] = df_train.description.map(lambda x: int(' андр' in x))
    df_train['мих'] = df_train.description.map(lambda x: int('миха' in x))
    df_train['одн'] = df_train.description.map(lambda x: int('одн' in x))
    df_train['('] = df_train.description.map(lambda x: int('(' in x))
    df_train[')'] = df_train.description.map(lambda x: int(')' in x))
    df_train['"'] = df_train.description.map(lambda x: int('"' in x))
    df_train['/'] = df_train.description.map(lambda x: int('/' in x))
    df_train[','] = df_train.description.map(lambda x: int(',' in x))
    df_train['Перм'] = df_train.description.map(lambda x: int('Перм' in x))
    df_train['мм'] = df_train.description.map(lambda x: int('мм' in x))
    df_train['худо'] = df_train.description.map(lambda x: int('худо' in x))
    df_train['кор'] = df_train.description.map(lambda x: int('кор' in x))
    df_train['муж'] = df_train.description.map(lambda x: int('муж' in x))
    df_train['групп'] = df_train.description.map(lambda x: int('групп' in x))
    df_train['рос'] = df_train.description.map(lambda x: int('рос' in x))
    df_train['рос'] = df_train.description.map(lambda x: int('рос' in x))
    df_train['рк'] = df_train.description.map(lambda x: int('рк' in x))
    df_train['пк'] = df_train.description.map(lambda x: int('пк' in x))
    df_train['го'] = df_train.description.map(lambda x: int('го' in x))
    df_train['вол'] = df_train.description.map(lambda x: int('вол' in x))
    df_train['ши'] = df_train.description.map(lambda x: int('ши' in x))
    df_train['бер'] = df_train.description.map(lambda x: int('бе' in x))
    df_train['ви'] = df_train.description.map(lambda x: int('ви' in x))
    df_train['ве'] = df_train.description.map(lambda x: int('веск' in x))
    df_train['i'] = df_train.description.map(lambda x: int('in' in x))
    df_train['па'] = df_train.description.map(lambda x: int('па' in x))
    df_train['портрет'] = df_train.description.map(lambda x: int('портрет' in x))
    df_train['масло'] = df_train.description.map(lambda x: int('масло' in x))
    ids = list(df_train.get('id'))
    X = df_train.drop(drop_fields, axis = 1)
    y = df_train.get('object_img')
    if not df_train.get('object_img') is None:
        y_indexes = pd.Series(list(df_train.get('object_img')), ids)
    else:
        y_indexes = None
    return X, y, y_indexes
X, y, y_indexes = load('train.csv', ["description","object_img",'id'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg = LinearRegression().fit(X, y)
pred=list(map(lambda x: abs(x), reg.predict(X_test)))
print("R2 score:", r2_score(y_test, pred))

test_X, test_y, test_y_indexes = load('data.csv', ["description",'id'])

p = list(map(lambda x: abs(x), reg.predict(test_X)))
data = {}
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for i, row in enumerate(reader, start=-1):
        if i == -1:
            continue
        if i < len(p):
            data[row[0]] = p[i]
        else:
            break
d = re.sub(r'[\-\:\.]', '_', re.sub(r'\s', 'T', str(datetime.datetime.today())))
with open(f'solution_{d}.csv', 'w', encoding='utf-8') as f:
    f.write('id,object_img\n')
    for key, value in list(data.items()):
        f.write(f'{key},{round(value)}\n')
print('Solutions created!')
