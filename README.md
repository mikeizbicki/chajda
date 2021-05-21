# chajda

Chajda (from the Korean 찾다, meaning to find) is a postgres extension and corresponding python library for highly multi-lingual full text search in postgres.
It has 3 primary goals:
1. support a large number of languages
1. correct, including maintain ACID-compliance
1. fast

Currently 58 languages are supported, which is significantly more than other full text search engines like:
ElasticSearch ([35](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html)),
Solr ([37](https://solr.apache.org/guide/8_0/language-analysis.html)),
Lucene ([37](https://lucene.apache.org/core/8_3_1/analyzers-common/index.html)),
Groonga (),
and native postgres (23).
In particular, native postgres has no support for full text search in Chinese, Japanese or Korean languages,
but the pspacy extension provides this functionality.

Pspacy adds support for these languages by using the popular [spacy](https://spacy.io/) python library to handle all language-specific parsing,
and then reusing Postgres's language agnostic full text search features (e.g. [GIST]()/[GIN]()/[RUM]() indexes) to perform the actual searching.
Spacy is under active development, and as more languages are added to spacy (and language support for the existing languages improves),
these improvements can be automatically reused within postgres.
No python code is called during a full text search query,
and so there is no query-time performance penalty for using a python library to parse the languages.

<!--
Pspacy requires postgres version >= 12 and python >= 3.6.
-->

1. [Installation](#Installation)
1. [Examples](#Examples)
1. [Perormance](#Performance)
1. [Limitations](#Limitations)

## Installation

Pspacy has a lot of dependencies, about 3GB in total.
This is due to the fact that each language has its own set of dependencies and models that must be downloaded and installed,
and some of these models are quite large.
Additionally many of the dependencies require careful tuning in order to get the correct, deterministic performance needed for use in a database.

Due to the complex nature of installing this library, it is recommended to use docker and the provided `Dockerfile`.
A working postgres 13 instance with pspacy installed can be created with the following commands:
```
$ git clone https://github.com/mikeizbicki/pspacy
$ cd pspacy
$ docker-compose up -d --build
```
You can then connect to the database via psql with the command
```
$ docker-compose exec pspacy_db psql
```

The `Dockerfile` internally uses the script `install_dependencies.sh` to download and install all of the needed dependencies.
If you cannot use docker for whatever reason, then you should be able to install with the commands
```
$ sh install_dependencies.sh
$ make
$ make install
```
Note that the versions of the installed libraries should be modified only with extreme care,
as modifying these dependencies can cause the same version of pspacy to parse text differently.
It is strongly recommended that you run the test cases to ensure that language parsing is working correctly:
```
$ python3 -m pytest     # ensures that the python correctly parses each language
$ make installcheck     # ensures that the library correcyly integrates with postgres
```

## Example

The following code loads the pspacy in a postgres database:
```
psql> CREATE LANGUAGE plpython3u;
psql> CREATE EXTENSION pspacy;
```
Installing the `plpython3u` language must be done before loading pspacy.

Next, we create a table and populate it with some multilingual text.
```
CREATE TABLE example (
    id SERIAL PRIMARY KEY,
    lang_iso TEXT,
    lang_name TEXT,
    doc TEXT
);

INSERT INTO example (lang_iso, lang_name, doc) VALUES
    ('ar', 'arabic', 'هذا هو جملتي التجريبية التي أقوم بوضعها في ترجمة Google لإنشاء حالات اختبار.'),
    ('da', 'danish', 'Dette er mit eksempel test sætning, som jeg lægger i Google Translate for at generere testsager.'),
    ('de', 'german', 'Dies ist mein Beispiel-Testsatz, den ich in Google Translate einfüge, um Testfälle zu generieren.'),
    ('el', 'greek', 'Αυτή είναι η παραδειγματική δοκιμαστική πρόταση που βάζω στη Μετάφραση Google για τη δημιουργία δοκιμαστικών περιπτώσεων.'),
    ('en', 'english', 'This is my example test sentence that I''m putting into Google Translate to generate test cases.'),
    ('es', 'spanish', 'Esta es mi oración de prueba de ejemplo que estoy poniendo en Google Translate para generar casos de prueba.'),
    ('fr', 'french', 'Ceci est mon exemple de phrase de test que je mets dans Google Translate pour générer des cas de test.'),
    ('it', 'italian', 'Questa è la mia frase di prova di esempio che sto inserendo in Google Translate per generare casi di test.'),
    ('ja', 'japanese', 'これは、テストケースを生成するためにGoogle翻訳に入力する私のテスト文の例です。'),
    ('ko', 'korean', '이것은 테스트 케이스를 생성하기 위해 Google Translate에 넣은 예제 테스트 문장입니다.'),
    ('lt', 'lithuanian', 'Tai yra mano bandomo sakinio pavyzdys, kurį dedu į „Google“ vertėją norėdamas sugeneruoti bandomuosius atvejus.'),
    ('nb', 'norwegian bokmal', 'Dette er mitt eksempel test setning som jeg legger i Google Translate for å generere testsaker.'),
    ('nl', 'dutch', 'Dit is mijn voorbeeldtestzin die ik in Google Translate zet om testcases te genereren.'),
    ('pl', 'polish', 'To jest moje przykładowe zdanie testowe, które umieszczam w Tłumaczu Google, aby wygenerować przypadki testowe.'),
    ('pt', 'portuguese', 'Esta é a minha frase de teste de exemplo que estou colocando no Google Translate para gerar casos de teste.'),
    ('ro', 'romanian', 'Acesta este exemplul meu de propoziție pe care îl introduc în Google Translate pentru a genera cazuri de testare.'),
    ('ru', 'russian', 'Это мое примерное тестовое предложение, которое я помещаю в Google Translate для создания тестовых случаев.'),
    ('uk', 'ukranian', 'Це мій приклад тестового речення, яке я вкладаю в Google Translate для створення тестових випадків.'),
    ('zh', 'chinese', '这是我输入Google Translate生成测试用例的示例测试语句。'),
    ('zh', 'chinese', '這是我輸入Google Translate生成測試用例的示例測試語句。'),
    ('af', 'afrikaans', 'Dit is my voorbeeldse toetssin wat ek in Google Translate plaas om toetsgevalle te genereer.'),
    ('sq', 'albanian', 'Kjo është fjalia ime e provës që po e vë në Google Translate për të gjeneruar raste testimi.'),
    ('hy', 'armenian', 'Սա իմ օրինակելի թեստային նախադասությունն է, որը ես դնում եմ Google Translate- ում ՝ թեստային դեպքեր առաջացնելու համար:'),
    ('eu', 'basque', 'Hau da Google Translate bertsioan jartzen ari naizen testeko esaldia. Test kasuak sortzeko.'),
    ('bn', 'bengali', 'এটি আমার উদাহরণ পরীক্ষার বাক্য যা পরীক্ষার কেসগুলি উত্পন্ন করার জন্য আমি গুগল অনুবাদে রাখছি।'),
    ('bg', 'bulgarian', 'Това е моето примерно изпитателно изречение, което вкарвам в Google Translate, за да генерирам тестови случаи.'),
    ('ca', 'catalan', 'Aquest és el meu exemple de frase de prova que vaig publicant a Google Translate per generar casos de prova.'),
    ('hr', 'croatian', 'Ovo je moja testna rečenica koju stavljam u Google Translate za generiranje testnih slučajeva.'),
    ('cs', 'czech', 'Toto je můj příklad zkušební věty, kterou vkládám do Překladače Google, abych generoval testovací případy.'),
    ('et', 'estonian', 'See on minu näidislause, mille panen testversioonide genereerimiseks Google''i tõlki.'),
    ('fi', 'finnish', 'Tämä on esimerkki testilauseestani, jonka laitan Google Translate -sovellukseen testitapausten luomiseksi.'),
    ('gu', 'gujarati', 'આ મારું ઉદાહરણ પરીક્ષણ વાક્ય છે કે જે હું પરીક્ષણનાં કેસો પેદા કરવા માટે Google અનુવાદમાં મૂકી રહ્યો છું.'),
    ('he', 'hebrew', 'זהו משפט המבחן הדוגמא שלי שאני מכניס ל- Google Translate כדי ליצור מקרי מבחן.'),
    ('hi', 'hindi', 'यह मेरा उदाहरण परीक्षण वाक्य है जो मैं परीक्षण मामलों को उत्पन्न करने के लिए Google अनुवाद में डाल रहा हूं।'),
    ('hu', 'hungarian', 'Ez a példakénti tesztmondat, amelyet a Google Fordítóba teszteket hozok létre.'),
    ('is', 'icelandic', 'Þetta er dæmi setningarpróf sem ég set í Google Translate til að búa til prófatilvik.'),
    ('id', 'indonesian', 'Ini adalah contoh kalimat pengujian yang saya masukkan ke Google Terjemahan untuk menghasilkan kasus uji.'),
    ('ga', 'irish', 'Is í seo mo phianbhreith tástála samplach atá á cur agam ar Google Translate chun cásanna tástála a ghiniúint.'),
    ('kn', 'kannada', 'ಪರೀಕ್ಷಾ ಪ್ರಕರಣಗಳನ್ನು ರಚಿಸಲು ನಾನು Google ಅನುವಾದಕ್ಕೆ ಹಾಕುತ್ತಿರುವ ನನ್ನ ಉದಾಹರಣೆ ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಇದು.'),
    ('lv', 'latvian', 'Šis ir mans testa teikuma piemērs, kuru es ievietoju Google Translate, lai ģenerētu testa gadījumus.'),
    ('lb', 'luxembourgish', 'Dëst ass mäi Beispill Testsaz deen ech a Google Translate setzen fir Testfäll ze generéieren.'),
    ('ml', 'malayalam', 'ടെസ്റ്റ് കേസുകൾ സൃഷ്ടിക്കുന്നതിനായി ഞാൻ Google വിവർത്തനത്തിലേക്ക് ഇടുന്ന എന്റെ ഉദാഹരണ പരീക്ഷണ വാക്യമാണിത്.'),
    ('mr', 'marathi', 'हे माझे उदाहरण चाचणी वाक्य आहे जे मी चाचणी प्रकरणे व्युत्पन्न करण्यासाठी Google भाषांतर मध्ये टाकत आहे.'),
    ('fa', 'farsi', 'این جمله آزمایشی نمونه ای است که من برای تولید موارد آزمایشی در Google Translate قرار می دهم.'),
    ('sr', 'sanskrit', 'Ово је мој пример тестне реченице коју стављам у Гоогле Транслате да бих генерисао тест случајеве.'),
    ('si', 'sinhala', 'පරීක්ෂණ අවස්ථා උත්පාදනය කිරීම සඳහා මම ගූගල් පරිවර්තනයට ඇතුළත් කරන මගේ උදාහරණ පරීක්ෂණ වාක්‍යය මෙයයි.'),
    ('sk', 'slovak', 'Toto je môj príklad testovacej vety, ktorú vkladám do služby Prekladač Google, aby som vygeneroval testovacie prípady.'),
    ('sl', 'slovenian', 'To je moj primer testnega stavka, ki ga vstavljam v Google Translate za ustvarjanje preskusnih primerov.'),
    ('sv', 'swedish', 'Detta är mitt exempel på testmeningen som jag lägger in i Google Translate för att generera testfall.'),
    ('tl', 'tagalog', 'Ito ang aking halimbawang pangungusap na pagsubok na inilalagay ko sa Google Translate upang makabuo ng mga kaso ng pagsubok.'),
    ('ta', 'tamil', 'சோதனை நிகழ்வுகளை உருவாக்க நான் Google மொழிபெயர்ப்பில் இடும் எனது எடுத்துக்காட்டு சோதனை வாக்கியம் இது.'),
    ('tt', 'tatar', 'Бу минем сынау үрнәкләрен ясау өчен Google Translate'ка куйган сынау җөмләсе.'),
    ('te', 'telugu', 'పరీక్ష కేసులను రూపొందించడానికి నేను గూగుల్ ట్రాన్స్‌లేట్‌లో పెడుతున్న నా ఉదాహరణ పరీక్ష వాక్యం ఇది.'),
    ('th', 'thai', 'นี่คือประโยคทดสอบตัวอย่างของฉันที่ฉันใส่ลงใน Google Translate เพื่อสร้างกรณีทดสอบ'),
    ('tr', 'turkish', 'Bu, test senaryoları oluşturmak için Google Çeviri'ye koyduğum örnek test cümledim.'),
    ('ur', 'urdu', 'یہ میرا مثال آزمائشی جملہ ہے جو میں گوگل کے ترجمے میں ٹیسٹ کے معاملات پیدا کرنے کے لئے ڈال رہا ہوں۔'),
    ('vi', 'vietnamese', 'Đây là câu kiểm tra mẫu của tôi mà tôi đang đưa vào Google Dịch để tạo các trường hợp kiểm tra.'),
    ('yo', 'yoruba', 'Eyi ni idajọ idanwo mi ti Mo n fi sinu Tumọ-ọrọ Google lati ṣe agbekalẹ awọn ọran idanwo.')
    ;
```
Postgres uses two types to perform full text search.
The first is `tsvector`, which represents the documents being searched over.

### tsvector

The following examples show the results of calling `pspacy_tsvector` on English, Spanish, Korean, and Chinese texts.
```
psql> select spacy_tsvector('en', $$This is my example test sentence that I'm putting into Google Translate to generate test cases.$$);
                                              spacy_tsvector                                              
----------------------------------------------------------------------------------------------------------
 'be':9 'case':17 'example':4 'generate':15 'google':12 'putt':10 'sentence':6 'test':5,16 'translate':13
```
```
psql> select spacy_tsvector('es', $$Esta es mi oración de prueba de ejemplo que estoy poniendo en Google Translate para generar casos de prueba.$$);
                                     spacy_tsvector                                     
----------------------------------------------------------------------------------------
 'caso':17 'generar':16 'google':13 'oración':4 'poner':11 'probar':6,19 'translate':14
```
```
psql> select spacy_tsvector('ko', $$이것은 테스트 케이스를 생성하기 위해 Google Translate에 넣은 예제 테스트 문장입니다.$$);
                                              spacy_tsvector                                               
-----------------------------------------------------------------------------------------------------------
 'google':10 'translate':11 '넣':13 '문장':17 '생성':6 '예제':15 '위하':9 '이':18 '케이스':4 '테스트':3,16
```
```
psql> select spacy_tsvector('zh', $$這是我輸入Google Translate生成測試用例的示例測試語句。$$);
                                        spacy_tsvector                                        
----------------------------------------------------------------------------------------------
 'google':4 'translate':5 '測試':7,11 '生成':6 '用例':8 '示例':10 '語句':12 '輸入':3 '這是':1
```

### tsquery

The second import type for full text search is `tsquery`, which represents the query doing the search.
The key difference between a `tsquery` and a `tsvector` is that words in a `tsquery` must be joined together using operators like `&`, `|`, and `<->`.

### generating indexes

There are two ways to generate indexes.
First, you can generate a giant index for all languages:
```
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector(lang_iso, doc));
```
Or you can generate separate partial indexes for each language:
```
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector('en', doc)) WHERE lang_iso='en';
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector('es', doc)) WHERE lang_iso='es';
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector('ko', doc)) WHERE lang_iso='ko';
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector('vi', doc)) WHERE lang_iso='vi';
CREATE INDEX example_idx ON example USING GIN(spacy_tsvector('zh', doc)) WHERE lang_iso='zh';
-- etc.
```
In terms of disk usage, both methods are the same.
The runtime performance of both methods differs, however.
Consider the problem of searching for the string `Google Translate` in the `example` table using the query
```
SELECT count(*)
FROM example
WHERE spacy_tsvector(lang_iso, doc) @@ spacy_tsquery(lang_iso, 'Google <-> Translate');
```
versus a language specific query like
```
SELECT count(*)
FROM example
WHERE spacy_tsvector(lang_iso, doc) @@ spacy_tsquery(lang_iso, 'Google <-> Translate')
  AND lang_iso='en';
```
The former query will be fastest with the single index, and the later query will be fastest with separate indexes.
The reason for this is that:
1. The former query with the single index will only have to scan a single index;
   the latter query will have to scan many indexes, and the query planner is likely to even choose to do a sequential scan instead due to the large number of indexes.
1. The later query with the single index will have a large posting list that must be filtered down,
   and each entry must inspect a page on the heap to check the `lang='en'` condition;
   using the multiple index strategy, then the posting list will be automatically filtered for us.

## Performance

Each language supported by spacy takes a different amount of time to parse,
but they all parse fast enough that pspacy can be used in realtime search applications (such as web search).

The benchmarks below measure the runtime of the following code
```
lemmatize('This is my example test sentence that I'm putting into Google Translate to generate test cases.')
```
with the string having been translated into the appropriate language.
The most expensive languages (Russian and Japanese) take 1-3 milliseconds to parse the input text,
but most languages parse in under 100 microseconds.

The benchmark command and results are shown below.
```
$ python3 -m pytest tests/test_bench.py --benchmark-columns=mean,stddev,max

--------------------------------- benchmark: 58 tests ---------------------------------
Name (time in us)              Mean              StdDev                   Max
---------------------------------------------------------------------------------------
test__lemmatize[tr]         12.8250 (1.0)        1.7303 (1.80)         90.0570 (2.19)
test__lemmatize[de]         20.1877 (1.57)       3.3317 (3.46)        167.8197 (4.08)
test__lemmatize[et]         21.2650 (1.66)       8.4457 (8.78)        459.5686 (11.18)
test__lemmatize[fi]         23.5396 (1.84)       1.8059 (1.88)        101.1435 (2.46)
test__lemmatize[sl]         23.7444 (1.85)       0.9619 (1.0)          41.1132 (1.0)
test__lemmatize[hu]         25.5053 (1.99)       2.5313 (2.63)        122.7977 (2.99)
test__lemmatize[pl]         26.0923 (2.03)       1.0453 (1.09)         45.0118 (1.09)
test__lemmatize[cs]         28.2501 (2.20)       1.4473 (1.50)         48.9857 (1.19)
test__lemmatize[sk]         28.7432 (2.24)       2.2361 (2.32)         94.5544 (2.30)
test__lemmatize[hy]         32.9398 (2.57)       2.9619 (3.08)        106.0106 (2.58)
test__lemmatize[bg]         33.2582 (2.59)       1.0792 (1.12)         51.4342 (1.25)
test__lemmatize[lv]         34.5079 (2.69)       2.2948 (2.39)        101.9286 (2.48)
test__lemmatize[da]         35.2996 (2.75)       2.9289 (3.04)        126.3991 (3.07)
test__lemmatize[lt]         35.3593 (2.76)       2.4452 (2.54)        108.5745 (2.64)
test__lemmatize[af]         43.2382 (3.37)       1.3490 (1.40)         62.5933 (1.52)
test__lemmatize[nl]         47.1462 (3.68)       3.3799 (3.51)        130.8154 (3.18)
test__lemmatize[tt]         51.0332 (3.98)       2.3669 (2.46)        118.6011 (2.88)
test__lemmatize[eu]         52.7648 (4.11)       2.5099 (2.61)         80.1897 (1.95)
test__lemmatize[ar]         54.5780 (4.26)       2.1365 (2.22)         87.2863 (2.12)
test__lemmatize[kn]         57.8738 (4.51)       2.9936 (3.11)         93.1127 (2.26)
test__lemmatize[bn]         58.3293 (4.55)       2.5854 (2.69)        129.2853 (3.14)
test__lemmatize[is]         61.0298 (4.76)       1.5998 (1.66)         86.6195 (2.11)
test__lemmatize[ta]         61.2081 (4.77)       1.5594 (1.62)         93.7460 (2.28)
test__lemmatize[en]         61.6253 (4.81)       1.8679 (1.94)         87.8060 (2.14)
test__lemmatize[el]         62.2469 (4.85)       3.8714 (4.02)        132.4899 (3.22)
test__lemmatize[nb]         63.9406 (4.99)       2.3149 (2.41)         95.7521 (2.33)
test__lemmatize[ro]         64.3471 (5.02)       4.7861 (4.98)        151.9015 (3.69)
test__lemmatize[tl]         65.2916 (5.09)       3.5376 (3.68)        140.1966 (3.41)
test__lemmatize[fa]         65.4769 (5.11)       3.0965 (3.22)        144.6763 (3.52)
test__lemmatize[te]         66.9424 (5.22)       2.7621 (2.87)        141.7072 (3.45)
test__lemmatize[es]         67.1596 (5.24)       3.8258 (3.98)        154.5437 (3.76)
test__lemmatize[he]         67.2693 (5.25)       3.0231 (3.14)        146.3983 (3.56)
test__lemmatize[hi]         68.4003 (5.33)       1.9675 (2.05)        103.4522 (2.52)
test__lemmatize[lb]         68.5446 (5.34)       2.6876 (2.79)        149.3199 (3.63)
test__lemmatize[mr]         69.1496 (5.39)       2.8065 (2.92)        142.3331 (3.46)
test__lemmatize[ga]         70.8458 (5.52)       2.6386 (2.74)        143.5317 (3.49)
test__lemmatize[sq]         71.1295 (5.55)       2.8590 (2.97)        144.9529 (3.53)
test__lemmatize[fr]         71.8590 (5.60)       2.6049 (2.71)        103.8415 (2.53)
test__lemmatize[ca]         73.2512 (5.71)       3.0663 (3.19)        165.1170 (4.02)
test__lemmatize[sr]         73.7751 (5.75)       2.1377 (2.22)        112.6193 (2.74)
test__lemmatize[hr]         74.0826 (5.78)       2.9661 (3.08)        150.8920 (3.67)
test__lemmatize[it]         74.1341 (5.78)       3.1757 (3.30)        148.9064 (3.62)
test__lemmatize[pt]         75.2631 (5.87)       2.8578 (2.97)        152.6885 (3.71)
test__lemmatize[si]         77.8949 (6.07)       4.7639 (4.95)        159.7954 (3.89)
test__lemmatize[sv]         81.0350 (6.32)     591.2129 (614.64)   20,832.1577 (506.70)
test__lemmatize[yo]         87.6180 (6.83)       7.6483 (7.95)        168.8404 (4.11)
test__lemmatize[ur]        107.3869 (8.37)       3.7989 (3.95)        177.4421 (4.32)
test__lemmatize[ml]        160.2818 (12.50)      5.9297 (6.16)        248.4377 (6.04)
test__lemmatize[gu]        174.7602 (13.63)      8.5273 (8.87)        263.1675 (6.40)
test__lemmatize[zh0]       227.7281 (17.76)      6.9270 (7.20)        310.1844 (7.54)
test__lemmatize[th]        283.6743 (22.12)      6.9012 (7.17)        347.7000 (8.46)
test__lemmatize[zh1]       322.9686 (25.18)      7.9270 (8.24)        422.2831 (10.27)
test__lemmatize[ko]        352.1828 (27.46)      9.0914 (9.45)        456.9618 (11.11)
test__lemmatize[vi]        602.6251 (46.99)     30.9842 (32.21)       655.8737 (15.95)
test__lemmatize[id]        644.1380 (50.23)     10.0530 (10.45)       759.1676 (18.47)
test__lemmatize[uk]        893.7093 (69.69)     25.8180 (26.84)     1,005.0004 (24.44)
test__lemmatize[ru]      1,088.2365 (84.85)     38.0368 (39.54)     1,220.1052 (29.68)
test__lemmatize[ja]      2,477.5590 (193.18)    79.5288 (82.68)     3,381.9266 (82.26)
---------------------------------------------------------------------------------------
```
The ISO code of the language being tested is shown in square brackets on the left.
Chinese has two different language codes (`zh0` and `zh1`) corresponding to the traditional and simplified results of Google translate.

Note that due to the lazy loading of language models, the first time you use a language will be significantly more expensive unless you explicitly call `load_language` first.

# Limitations

Dedicated full text search engines like ElasticSearch/Lucene/Solr/Groonga currently provide a number of features that pspacy does not.
