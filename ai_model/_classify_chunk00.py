# -*- coding: utf-8 -*-
import pandas as pd
import re

df = pd.read_csv(r"F:\WomenSafetyAI_Complete_Project_Structure\dataset\review\chunks\chunk_00.csv")

def norm(s):
    return str(s).lower()

# All patterns require a LEADING \b (word start) to avoid matching common
# Tamil/Telugu grammatical suffixes (e.g. "-thu", "-nai") that appear at the
# END of ordinary words. Trailing \b added only where safe.

VULGAR = [
    r"\bnaa+i\b", r"\bnaa+ya", r"నాయి", r"நாய்", r"ನಾಯಿ", r"\bkukka\b", r"కుక్క",
    r"\bgadida", r"గాడిద", r"\bpandi\b", r"பன்றி", r"హంది", r"\bhandi\b",
    r"\bkazhudai", r"கழுதை",
    r"\bpunda", r"\bpundai", r"பூந்தை", r"పుకు", r"\bpuk\b", r"\bpuku",
    r"\bsunni\b", r"thevidiy", r"thevudiy", r"tavediy", r"thavediy", r"tavidiy",
    r"தேவிடிய", r"தேவுடிய", r"ದೇವುಡಿಯ",
    r"\btullu", r"ತುಲ್ಲು",
    r"\bkoothi", r"கூத்தி", r"\blanja", r"లంజ",
    r"\blavda", r"\blowda", r"ಲೌಡ", r"\bmodda", r"మొడ్డ", r"\bgudda\b", r"గుడ్డ",
    r"\boombu", r"ఊంబు", r"\botha\b", r"ஒத்த", r"\bdenge", r"\bdengu", r"దెంగు",
    r"\bsule\b", r"ಸುಳೆ", r"\bsulemakale", r"\bbosdi", r"\bbhosdi",
    r"\bkojja", r"కొజ్జ", r"\bseruppadi", r"செருப்படி", r"\bmayiru", r"மயிரு", r"మయిరు",
    r"\bmaire\b",
    r"\berri", r"\bpichi\b", r"వెధవ", r"\bvedava", r"\bloose\b",
    r"\bbokka\b", r"బొక్క", r"\bgommala", r"\bthu+\b",
    r"\bpukka\b", r"\btrash\b", r"\bstupid\b", r"\bidiot\b", r"\bnonsense\b",
    r"\bshut\s*up\b", r"\bpig\b", r"\bdog\b", r"\bkeeltha",
    r"\bgandu\b", r"\bkevalama", r"\bpottai\b", r"drink.*urine", r"urine.*vangi",
]

MISOGYNY = [
    r"women should", r"pombala.*veli.*poga", r"pen.*veliya varak",
    r"never trust.*women", r"\bfeminist", r"பெண்கள்.*வெளிய",
    r"girls should stay", r"ఆడవాళ్ళు.*బయట", r"ఆడవాళ్ళను.*నమ్మ",
    r"pundaingala ellarukum", r"othanunga.*akka",
]

THREAT = [
    r"\bkill you\b", r"\bi will kill\b", r"unnai kollu", r"uyir edu", r"champutha",
    r"\bchastha\b", r"champesta", r"கொலை செய்", r"prananni theestha",
    r"othanunga",
]

SEXUAL = [
    r"\bsexy\b", r"\bboobs\b", r"figure.*hot", r"body.*hot", r"\bnude\b",
    r"item.*bomb", r"pilla puk", r"pundai.*girl",
    r"\bvirgin\?", r"\bassault\b",
]

def find_match(patterns, text):
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return p
    return None

rows = []
for _, r in df.iterrows():
    cid = r['comment_id']
    cat = r['category']
    text = norm(r['comment'])

    m_vulgar = find_match(VULGAR, text)
    m_miso = find_match(MISOGYNY, text)
    m_threat = find_match(THREAT, text)
    m_sexual = find_match(SEXUAL, text)

    verdict = None
    corrected = ""
    reason = ""

    if cat == "Hate Speech":
        if m_miso:
            verdict, reason = "Correct", "explicit misogynistic generalization"
        elif m_threat:
            verdict, corrected, reason = "Wrong", "Threat", "contains direct threat/violence language"
        elif m_sexual:
            verdict, corrected, reason = "Wrong", "Sexual Harassment", "sexualized comment about a person"
        elif m_vulgar:
            verdict, corrected, reason = "Wrong", "Offensive", "vulgar insult but no group/gender hate target"
        else:
            verdict, corrected, reason = "Wrong", "Safe", "movie/actor/political comment, no hate content"
    elif cat == "Offensive":
        if m_miso:
            verdict, corrected, reason = "Wrong", "Hate Speech", "explicit misogynistic generalization"
        elif m_threat:
            verdict, corrected, reason = "Wrong", "Threat", "contains direct threat/violence language"
        elif m_sexual:
            verdict, corrected, reason = "Wrong", "Sexual Harassment", "sexualized comment about a person"
        elif m_vulgar:
            verdict, reason = "Correct", "vulgar/rude insult present"
        else:
            verdict, corrected, reason = "Wrong", "Safe", "mild/neutral comment, no insult"
    else:
        verdict, reason = "Unsure", "unexpected category"

    rows.append({"comment_id": cid, "verdict": verdict, "corrected_category": corrected, "reason": reason,
                 "_cat": cat, "_text": r['comment'],
                 "_m_vulgar": m_vulgar or "", "_m_miso": m_miso or "",
                 "_m_threat": m_threat or "", "_m_sexual": m_sexual or ""})

out = pd.DataFrame(rows)
print(out['verdict'].value_counts())
print(out.groupby(['_cat','verdict']).size())
print(out['corrected_category'].value_counts(dropna=False))
out.to_csv(r"F:\WomenSafetyAI_Complete_Project_Structure\dataset\review\chunks\_draft_verdict_00.csv", index=False, encoding="utf-8-sig")
print("done")
