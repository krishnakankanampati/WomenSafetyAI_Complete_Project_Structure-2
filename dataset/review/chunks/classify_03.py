# -*- coding: utf-8 -*-
import pandas as pd
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('chunk_03.csv')

def has(text, words):
    t = text.lower()
    return any(w.lower() in t for w in words)

# ---------- Word lists (romanized + native script, covers Tamil/Telugu/Kannada/Tanglish/Kanglish) ----------

SEXUAL_HARASSMENT_WORDS = [
    'send ur address', 'send your address', 'ur address plz',
    'figure ki salute', 'sanka lo body', 'sankalo body', 'ukka posthu',
    'ತುಲ್ಲ್ ನೋಡ್ಬೇಕ್', 'plz send ur address',
]

THREAT_WORDS = [
    'i will kill you', 'naanu unnai kolluven', 'unnai kollividuven',
]

# caste/community/religion hate markers - substrings that when present indicate group-targeted hate
CASTE_RELIGION_HATE_MARKERS = [
    'jaadhi veri', 'jaathi veri', 'jaathi thimira', 'saathi veri',
    'சாதி வெறி', 'ஜாதி வெறி', 'jathi veri', 'jaathi veri pudicha',
    'சாதி நாய்', 'ஜாதி நாய்', 'jaathiya naaikal', 'நாய்களுக்கு',
    'sc thevdya', 'sc echai', 'sc s echai',
    'தலித் அல்லா', 'சாதிகார தேவிடியா', 'sakkiliyara',
    'பறையர', 'பறையன', 'parayan', 'parayans',
    'பார்ப்பன', 'brahmin ', 'ಬ್ರಾಹ್ಮಣ',
    'கிறிஸ்துவ மிஷனரிக', 'christian missionar',
    'ஜிஹாதி', 'jihadi',
    'வன்னியர்', 'தேவர் புண்ட', 'thevar punda', 'vanniyar kovandar',
    'குல சத்திரிய', 'சாதி பெயர',
    'jaadhi kooothi', 'jaathi koothinu', 'jaathi koothi',
    'thevdiya paiya enaku mathamay',
    'kula pechhe', 'low caste pasangala',
]

MISOGYNY_GENERALIZATION_MARKERS = [
    'ಮೋಸ ಮಾಡಿ ಹೋಗುವ ಹುಡುಗಿರಿಗೆ', # girls who cheat deserve this lesson
    'women ki respect ledhu',
]

# vulgar / sexual slur words -> generally Offensive (personal insult), unless combined with the above hate markers
VULGAR_WORDS = [
    'lawda','lawde','lauda','lowda','lowde','loda','lund','lodu','laude','loude','louda','lowdre','lowdan',
    'gaandu','gandu','gaanduge','gaandla','gandla','gaandlu','gandu galge',
    'thulla','thulle','thulli','tulli','tullu','tulla','ತುಲ್ಲ','ತುಲ್ಲು','ತುಲ್ಲಗೆ','ತುಲ್ಲಗ',
    'modda','moddha','ಮೊದ','puku','pukku','poola','poolu','pundai','punda','ಪುಂಡೈ',
    'koothi','koothii','kootti','kuthi','கூத்தி','பூல்','சுன்னி','sunni',
    'boothulu','bhoothulu','boothu','bootu',
    'sule','soole','soolemagane','soolimagane','soolemakalu','soolemaklu','ಸೂಳೆ','ಸುಳೆ',
    'lanja','lanjakodaka','ಲಂಜ','thevidiya','thevdiya','thevidya','தேவிடியா','தேவடியா',
    'bosudi','bosdi','bosudike','ranku mundalu','mundalu','ಮುಂಡೆ','ರಂಡಿ','randi','rande','ranku',
    'dengu','dengav','dengudu','dengithe','dengesi','dengorige','dengay','deng',
    'gudda','ಗುದ್ದ','tunni','tunne','ಗಬ್ಬು ವಾಸ್ನಿ',
    'naamard','naamarda','ನಾಮರ್ದ','ಹಾದರಗಿತ್ತಿ','hadaragitti',
    'chakka munde','ಚಕ್ಕ ಮುಂಡೆ','ஊம்பு','ஊம்பிக',
    'sooolemagane','sulemakalla','ಬೋಳಿಮಗ','boli maga','bolimagan',
    'kanapundai','pora pilla puka','naaida','poonda','poondaingala','punda pasangala',
    'lucchan','ಲುಚ್ಛನ್',
]

# rude / mocking insult words (no explicit vulgar slur) -> Offensive
RUDE_INSULT_WORDS = [
    'erripu','erripuku','erripukulu','yerripu',
    'kevalam','kevala','trash','stupid','idiot','moron',
    'pichodu','picha na kodaka','pichi','manda battharam','erri pu gadu','ottu',
    'komaali','komali','loose ','loosu','loosa','mokka','mokkaya','mokkai',
    'muttal','muttalukal','moola pathiram','mairu','mairukku','maireeti',
    'kadupa','naaikaluku','naaiku','naaikale','naaikalukum','naaidu',
    'pundaingala','poori','porambo','porambol','vedhava','vedava','vetakal',
    'chutiya','chuthiya','ಚುತಿಯಾ',
    'thayoli','thayolinga','thevidiya kandaraoli','arivu kettavana',
    'kolla gand','kola gand','kolla gaand',
    'trash da','waste gadu','worthless',
]

def classify(row):
    text = str(row['comment'])
    tl = text.lower()

    # Sexual harassment - explicit sexualized remark about a person's body / solicitation
    if has(tl, SEXUAL_HARASSMENT_WORDS):
        return 'Sexual Harassment'

    # explicit misogynistic generalization endorsing control/violence toward women as a group
    if has(text, MISOGYNY_GENERALIZATION_MARKERS):
        return 'Hate Speech'

    # caste / religion / community targeted hate
    if has(tl, CASTE_RELIGION_HATE_MARKERS):
        return 'Hate Speech'

    # explicit direct threats
    if has(tl, THREAT_WORDS):
        return 'Threat'

    # vulgar slur words -> offensive (personal vulgar abuse, not group-targeted)
    if has(tl, VULGAR_WORDS):
        return 'Offensive'

    if has(tl, RUDE_INSULT_WORDS):
        return 'Offensive'

    return 'Safe'

df['rule_verdict'] = df.apply(classify, axis=1)
print(df['rule_verdict'].value_counts())
print()
print(pd.crosstab(df['category'], df['rule_verdict']))
df.to_csv('chunk_03_ruled.csv', index=False, encoding='utf-8-sig')
