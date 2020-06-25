import pytest
from postgresql_spacy import lemmatize

tests = [
    ( "ar", "هذا هو جملتي التجريبية التي أقوم بوضعها في ترجمة Google لإنشاء حالات اختبار.", "هذا هو جملتي التجريبية التي أقوم بوضعها في ترجمة google لإنشاء حالات اختبار ."),
    ( "da", "Dette er mit eksempel test sætning, som jeg lægger i Google Translate for at generere testsager.", "denne være min eksempel test sætning , som jeg lægge i google translate for at generere testsager ."),
    ( "de", "Dies ist mein Beispiel-Testsatz, den ich in Google Translate einfüge, um Testfälle zu generieren.", "dies sein meinen beispiel-testsatz , der ich in google translate einfügen , um testfälle zu generieren ."),
    ( "el", "Αυτή είναι η παραδειγματική δοκιμαστική πρόταση που βάζω στη Μετάφραση Google για τη δημιουργία δοκιμαστικών περιπτώσεων.", "αυτή είναι η παραδειγματική δοκιμαστική πρόταση που βάζω στη μετάφραση google για τη δημιουργία δοκιμαστικών περιπτώσεων ."),
    ( "en", "This is my example test sentence that I'm putting into Google Translate to generate test cases.", "this be my example test sentence that -pron- be putt into google translate to generate test case ." ),
    ( "es", "Esta es mi oración de prueba de ejemplo que estoy poniendo en Google Translate para generar casos de prueba.", "este ser mi oración de probar de ejemplo que estar poner en google translate parir generar caso de probar ."),
    ( "fr", "Ceci est mon exemple de phrase de test que je mets dans Google Translate pour générer des cas de test.", "ceci est mon exemple de phrase de test que je mets dans google translate pour générer un cas de test ."),
    ( "it", "Questa è la mia frase di prova di esempio che sto inserendo in Google Translate per generare casi di test.", "questo essere la mio frase di provare di esempio che stare inserire in google translate per generare caso di test ."),
    ( "ja", "これは、テストケースを生成するためにGoogle翻訳に入力する私のテスト文の例です。", "これ は 、 テスト ケース を 生成 する ため に google 翻訳 に 入力 する 私 の テスト 文 の 例 です 。"),
    ( "ko", "이것은 테스트 케이스를 생성하기 위해 Google Translate에 넣은 예제 테스트 문장입니다.", "이거 은 테스트 케이스 를 생성 하 기 위하 google translate 에 넣 은 예제 테스트 문장 이 ."),
    ( "lt", "Tai yra mano bandomo sakinio pavyzdys, kurį dedu į „Google“ vertėją norėdamas sugeneruoti bandomuosius atvejus.", "tai irti manyti bandomo sakinys pavyzdys , kurį dėti į „ google “ vertėjas norėdamas sugeneruoti bandomas atvejis ."),
    ( "nb", "Dette er mitt eksempel test setning som jeg legger i Google Translate for å generere testsaker.", "dette er min eksempel test setning som jeg legger i google translate for å generere testsaker ."),
    ( "nl", "Dit is mijn voorbeeldtestzin die ik in Google Translate zet om testcases te genereren.", "dit is mijn voorbeeldtestzin die ik in google translate zet om testcase te genereren ."),
    ( "pl", "To jest moje przykładowe zdanie testowe, które umieszczam w Tłumaczu Google, aby wygenerować przypadki testowe.", "to jest moje przykładowe zdanie testowe , które umieszczam w tłumaczu google , aby wygenerować przypadki testowe ."),
    ( "pt", "Esta é a minha frase de teste de exemplo que estou colocando no Google Translate para gerar casos de teste.", "este ser o meu frase de testar de exemplo que estar colocar o google translate parir gerar caso de testar ."),
    ( "ro", "Acesta este exemplul meu de propoziție pe care îl introduc în Google Translate pentru a genera cazuri de testare.", "acesta fi exemplu meu de propoziție pe căra el introduce în google translate pentru avea genera caz de testare ."),
    ( "ru", "Это мое примерное тестовое предложение, которое я помещаю в Google Translate для создания тестовых случаев.", "это мое примерное тестовое предложение , которое я помещать в google translate для создания тестовых случай ."),
    ( "uk", "Це мій приклад тестового речення, яке я вкладаю в Google Translate для створення тестових випадків.", "це мій приклад тестового речення , яке я вкладати в google translate для створення тестових випадок ."),
    ( "zh", "这是我输入Google Translate生成测试用例的示例测试语句。", "这 是 我 输入 google translate 生成 测试用例 的 示例 测试 语句 。"), # simplified
    ( "zh", "這是我輸入Google Translate生成測試用例的示例測試語句。", "這是 我 輸入 google translate 生成 測試 用例 的 示例 測試 語句 。"), # traditional
    ( "af", "Dit is my voorbeeldse toetssin wat ek in Google Translate plaas om toetsgevalle te genereer.", "dit is my voorbeeldse toetssin wat ek in google translate plaas om toetsgevalle te genereer ."),
    ( "sq", "Kjo është fjalia ime e provës që po e vë në Google Translate për të gjeneruar raste testimi.", "kjo është fjalia ime e provës që po e vë në google translate për të gjeneruar raste testimi ."),
    ( "hy", "Սա իմ օրինակելի թեստային նախադասությունն է, որը ես դնում եմ Google Translate- ում ՝ թեստային դեպքեր առաջացնելու համար:", "սա իմ օրինակելի թեստային նախադասությունն է , որը ես դնում եմ google translate- ում ՝ թեստային դեպքեր առաջացնելու համար :"),
    ( "eu", "Hau da Google Translate bertsioan jartzen ari naizen testeko esaldia. Test kasuak sortzeko.", "hau da google translate bertsioan jartzen ari naizen testeko esaldia . test kasuak sortzeko ."),
    ( "bn", "এটি আমার উদাহরণ পরীক্ষার বাক্য যা পরীক্ষার কেসগুলি উত্পন্ন করার জন্য আমি গুগল অনুবাদে রাখছি।", "এটি আমার উদাহরণ পরীক্ষার বাক্য যা পরীক্ষার কেসগুলি উত্পন্ন করার জন্য আমি গুগল অনুবাদে রাখছি ।"),
    ( "bg", "Това е моето примерно изпитателно изречение, което вкарвам в Google Translate, за да генерирам тестови случаи.", "това е моето примерно изпитателно изречение , което вкарвам в google translate , за да генерирам тестови случаи ."),
    ( "ca", "Aquest és el meu exemple de frase de prova que vaig publicant a Google Translate per generar casos de prova.", "aquest ser ell meu exemple de frase de provar que anar publicar a google translate per generar cas de provar ."),
    ( "hr", "Ovo je moja testna rečenica koju stavljam u Google Translate za generiranje testnih slučajeva.", "ovaj biti moj testni rečenica koji stavljati u google translate za generiranje testni slučaj ."),
    ( "cs", "Toto je můj příklad zkušební věty, kterou vkládám do Překladače Google, abych generoval testovací případy.", "toto je můj příklad zkušební věty , kterou vkládám do překladače google , abych generoval testovací případy ."),
    ( "et", "See on minu näidislause, mille panen testversioonide genereerimiseks Google'i tõlki.", "see on minu näidislause , mille panen testversioonide genereerimiseks google'i tõlki ."),
    ( "fi", "Tämä on esimerkki testilauseestani, jonka laitan Google Translate -sovellukseen testitapausten luomiseksi.", "tämä on esimerkki testilauseestani , jonka laitan google translate -sovellukseen testitapausten luomiseksi ."),
    ( "gu", "આ મારું ઉદાહરણ પરીક્ષણ વાક્ય છે કે જે હું પરીક્ષણનાં કેસો પેદા કરવા માટે Google અનુવાદમાં મૂકી રહ્યો છું.", "આ મારું ઉદાહરણ પરીક્ષણ વાક્ય છે કે જે હું પરીક્ષણનાં કેસો પેદા કરવા માટે google અનુવાદમાં મૂકી રહ્યો છું."),
    ( "he", "זהו משפט המבחן הדוגמא שלי שאני מכניס ל- Google Translate כדי ליצור מקרי מבחן.", "זהו משפט המבחן הדוגמא שלי שאני מכניס ל- google translate כדי ליצור מקרי מבחן ."),
    ( "hi", "यह मेरा उदाहरण परीक्षण वाक्य है जो मैं परीक्षण मामलों को उत्पन्न करने के लिए Google अनुवाद में डाल रहा हूं।", "यह मेरा उदाहरण परीक्षण वाक्य है जो मैं परीक्षण मामलों को उत्पन्न करने के लिए google अनुवाद में डाल रहा हूं ।"),
    ( "hu", "Ez a példakénti tesztmondat, amelyet a Google Fordítóba teszteket hozok létre.", "ez a példakénti tesztmondat , amely a google fordítóba teszteket hoz lét ."),
    ( "is", "Þetta er dæmi setningarpróf sem ég set í Google Translate til að búa til prófatilvik.", "þetta er dæmi setningarpróf sem ég set í google translate til að búa til prófatilvik ."),
    ( "id", "Ini adalah contoh kalimat pengujian yang saya masukkan ke Google Terjemahan untuk menghasilkan kasus uji.", "ini adalah contoh kalimat uji yang saya masukkan ke google terjemah untuk hasil kasus uji ."),
    ( "ga", "Is í seo mo phianbhreith tástála samplach atá á cur agam ar Google Translate chun cásanna tástála a ghiniúint.", "is í seo mo phianbhreith tástála samplach atá á cur agam ar google translate chun cásanna tástála a ghiniúint ."),
    ( "kn", "ಪರೀಕ್ಷಾ ಪ್ರಕರಣಗಳನ್ನು ರಚಿಸಲು ನಾನು Google ಅನುವಾದಕ್ಕೆ ಹಾಕುತ್ತಿರುವ ನನ್ನ ಉದಾಹರಣೆ ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಇದು.", "ಪರೀಕ್ಷಾ ಪ್ರಕರಣಗಳನ್ನು ರಚಿಸಲು ನಾನು google ಅನುವಾದಕ್ಕೆ ಹಾಕುತ್ತಿರುವ ನನ್ನ ಉದಾಹರಣೆ ಪರೀಕ್ಷಾ ವಾಕ್ಯ ಇದು ."),
    ( "lv", "Šis ir mans testa teikuma piemērs, kuru es ievietoju Google Translate, lai ģenerētu testa gadījumus.", "šis ir mans testa teikuma piemērs , kuru es ievietoju google translate , lai ģenerētu testa gadījumus ."),
    # FIXME: No google translate ( "lij", "", ""),
    ( "lb", "Dëst ass mäi Beispill Testsaz deen ech a Google Translate setzen fir Testfäll ze generéieren.", "dëst ass mäin beispill testsaz deen ech an google translate setzen fir testfäll ze generéieren ."),
    ( "ml", "ടെസ്റ്റ് കേസുകൾ സൃഷ്ടിക്കുന്നതിനായി ഞാൻ Google വിവർത്തനത്തിലേക്ക് ഇടുന്ന എന്റെ ഉദാഹരണ പരീക്ഷണ വാക്യമാണിത്.", "ടെസ്റ്റ് കേസുകൾ സൃഷ്ടിക്കുന്നതിനായി ഞാൻ google വിവർത്തനത്തിലേക്ക് ഇടുന്ന എന്റെ ഉദാഹരണ പരീക്ഷണ വാക്യമാണിത്."),
    ( "mr", "हे माझे उदाहरण चाचणी वाक्य आहे जे मी चाचणी प्रकरणे व्युत्पन्न करण्यासाठी Google भाषांतर मध्ये टाकत आहे.", "हे माझे उदाहरण चाचणी वाक्य आहे जे मी चाचणी प्रकरणे व्युत्पन्न करण्यासाठी google भाषांतर मध्ये टाकत आहे ."),
    ( "fa", "این جمله آزمایشی نمونه ای است که من برای تولید موارد آزمایشی در Google Translate قرار می دهم.", "این جمله آزمایشی نمونه ای است که من برای تولید موارد آزمایشی در google translate قرار می دهم ."),
    ( "sr", "Ово је мој пример тестне реченице коју стављам у Гоогле Транслате да бих генерисао тест случајеве.", "овај бити мој пример тестни реченице који стављати у гоогле транслате дати бити генерисати тест случајеве ."),
    ( "si", "පරීක්ෂණ අවස්ථා උත්පාදනය කිරීම සඳහා මම ගූගල් පරිවර්තනයට ඇතුළත් කරන මගේ උදාහරණ පරීක්ෂණ වාක්‍යය මෙයයි.", "පරීක්ෂණ අවස්ථා උත්පාදනය කිරීම සඳහා මම ගූගල් පරිවර්තනයට ඇතුළත් කරන මගේ උදාහරණ පරීක්ෂණ වාක්‍යය මෙයයි ."),
    ( "sk", "Toto je môj príklad testovacej vety, ktorú vkladám do služby Prekladač Google, aby som vygeneroval testovacie prípady.", "toto je môj príklad testovacej vety , ktorú vkladám do služby prekladač google , aby som vygeneroval testovacie prípady ."),
    ( "sl", "To je moj primer testnega stavka, ki ga vstavljam v Google Translate za ustvarjanje preskusnih primerov.", "to je moj primer testnega stavka , ki ga vstavljam v google translate za ustvarjanje preskusnih primerov ."),
    ( "sv", "Detta är mitt exempel på testmeningen som jag lägger in i Google Translate för att generera testfall.", "denna vara jag exempel på testmeningen som jag lägga i i google translate föra att generera testfall ."),
    ( "tl", "Ito ang aking halimbawang pangungusap na pagsubok na inilalagay ko sa Google Translate upang makabuo ng mga kaso ng pagsubok.", "ito ang aking halimbawang pangungusap na pagsubok na inilalagay ko sa google translate upang makabuo ng mga kaso ng pagsubok ."),
    ( "ta", "சோதனை நிகழ்வுகளை உருவாக்க நான் Google மொழிபெயர்ப்பில் இடும் எனது எடுத்துக்காட்டு சோதனை வாக்கியம் இது.", "சோதனை நிகழ்வுகளை உருவாக்க நான் google மொழிபெயர்ப்பில் இடும் எனது எடுத்துக்காட்டு சோதனை வாக்கியம் இது ."),
    ( "tt", "Бу минем сынау үрнәкләрен ясау өчен Google Translate'ка куйган сынау җөмләсе.", "бу минем сынау үрнәкләрен ясау өчен google translate ' ка куйган сынау җөмләсе ."),
    ( "te", "పరీక్ష కేసులను రూపొందించడానికి నేను గూగుల్ ట్రాన్స్‌లేట్‌లో పెడుతున్న నా ఉదాహరణ పరీక్ష వాక్యం ఇది.", "పరీక్ష కేసులను రూపొందించడానికి నేను గూగుల్ ట్రాన్స్‌లేట్‌లో పెడుతున్న నా ఉదాహరణ పరీక్ష వాక్యం ఇది ."),
    ( "th", "นี่คือประโยคทดสอบตัวอย่างของฉันที่ฉันใส่ลงใน Google Translate เพื่อสร้างกรณีทดสอบ", "นี่ คือ ประโยค ทดสอบ ตัวอย่าง ของ ฉัน ที่ ฉัน ใส่ ลง ใน   google   translate   เพื่อ สร้าง กรณี ทดสอบ"),
    ( "tr", "Bu, test senaryoları oluşturmak için Google Çeviri'ye koyduğum örnek test cümledim.", "bu , test senaryo oluş için google çeviri'ye koy örnek test cümledim ."),
    ( "ur", "یہ میرا مثال آزمائشی جملہ ہے جو میں گوگل کے ترجمے میں ٹیسٹ کے معاملات پیدا کرنے کے لئے ڈال رہا ہوں۔", "میں میرا مثال آزمائشی جملہ ہونا جو میں گوگل کم ترجمہ میں ٹیسٹ کم معاملات پیدا کرنا کم لئے ڈالنا رہنا ہونا ۔"),
    ( "vi", "Đây là câu kiểm tra mẫu của tôi mà tôi đang đưa vào Google Dịch để tạo các trường hợp kiểm tra.", "đây là câu kiểm_tra mẫu của tôi mà tôi đang đưa vào google dịch để tạo các trường_hợp kiểm_tra ."),
    ( "yo", "Eyi ni idajọ idanwo mi ti Mo n fi sinu Tumọ-ọrọ Google lati ṣe agbekalẹ awọn ọran idanwo.", "eyi ni idajọ idanwo mi ti mo n fi sinu tumọ - ọrọ google lati ṣe agbekalẹ awọn ọran idanwo ."),
    ]

@pytest.mark.parametrize('lang, text, output', tests)
def test__lemmatize(lang, text, output):
    assert lemmatize(lang,text,no_special_chars=False) == output


problematic_strings = [
    ('ko', '2/13하루만 예약이 안되길래 이날 오는거 아닌가 예상해보고 바로 다음날로 예약했는데 나 선견지명있었군　( ͡° ͜ʖ ͡°)'),
    ]

@pytest.mark.parametrize('lang, text', problematic_strings)
def test__lemmatize_2(lang, text):
    lemmatize(lang,text)
