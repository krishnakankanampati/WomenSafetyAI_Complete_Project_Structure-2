"""
80 freshly-authored test messages: 7 languages x 5 categories.

None of these come from the dataset - they are written from scratch so the
result is not contaminated by anything the model saw in training. Sexual
Harassment is deliberately over-sampled (24 of 80) because that is the
suspected blind spot; every breakdown below is reported per-category so the
weighting does not hide behind a single headline number.

Parallel content across languages is intentional: several items are the same
sentence translated, so a label that changes with language is evidence of a
language shortcut rather than a content judgement.

Usage (run from the project root, one level above ai_model/):
    python -m ai_model.eval_language_matrix

Runs entirely on the local model - no API key and no network needed.
"""
import sys
from collections import defaultdict

import pandas as pd

from ai_model.predict import WomenSafetyPredictor

S, O, X, T, H = "Safe", "Offensive", "Sexual Harassment", "Threat", "Hate Speech"

DATA = [
    # ---------------- English ----------------
    ("English", S, "The concert last night was amazing, the crowd was so energetic"),
    ("English", S, "Can someone share the notes from today's lecture please"),
    ("English", O, "You are such an idiot, nobody wants to hear your stupid opinions"),
    ("English", O, "Shut up you worthless loser, go cry somewhere else"),
    ("English", X, "You looked so hot in that dress, send me a private pic"),
    ("English", X, "Why don't you come to my room tonight, no one will know"),
    ("English", X, "Nice smile, what's your number? we can meet alone somewhere"),
    ("English", T, "I have your address, wait and see what happens to you"),
    ("English", T, "Keep talking and I will break your legs when I find you"),
    ("English", H, "Women in this country should never be allowed to work, they belong at home"),
    ("English", H, "People from that community are all criminals by birth"),

    # ---------------- Telugu (native script) ----------------
    ("Telugu", S, "ఈ రోజు వాతావరణం చాలా బాగుంది, బయటకు వెళ్దాం"),
    ("Telugu", S, "మీ వంట రెసిపీ చాలా బాగుంది, థాంక్స్"),
    ("Telugu", O, "నువ్వు పనికిరాని వెధవ, నీ మాటలు ఎవరూ వినరు"),
    ("Telugu", O, "మూర్ఖుడా, నోరు మూసుకో"),
    ("Telugu", X, "నీ ఫోటో పంపు, ఎవరికీ చెప్పను"),
    ("Telugu", X, "నువ్వు చాలా అందంగా ఉన్నావు, ఒంటరిగా కలుద్దామా?"),
    ("Telugu", X, "నీ డ్రెస్ లో సూపర్ గా ఉన్నావ్, రాత్రి వీడియో కాల్ చెయ్యి"),
    ("Telugu", T, "నువ్వు ఎక్కడ ఉంటావో నాకు తెలుసు, జాగ్రత్త"),
    ("Telugu", T, "ఇంకోసారి మాట్లాడితే నిన్ను వదిలిపెట్టను"),
    ("Telugu", H, "ఆ కులం వాళ్ళంతా దొంగలు, వాళ్ళని నమ్మకూడదు"),
    ("Telugu", H, "ఆడవాళ్ళకి చదువు అవసరం లేదు, వాళ్ళు ఇంట్లోనే ఉండాలి"),

    # ---------------- Roman Telugu ----------------
    ("Roman Telugu", S, "ee cinema chala bagundi, malli chudalani undi"),
    ("Roman Telugu", S, "repu meeting ki time enti cheppandi"),
    ("Roman Telugu", O, "nuvvu oka pichi vedhava, nee matalu evaru vinaru"),
    ("Roman Telugu", O, "sigguleni manishi, nee mokham chudaleka chastunna"),
    ("Roman Telugu", X, "nee photo pampu, evarikkee cheppanu"),
    ("Roman Telugu", X, "chala andanga unnav, oka sari ontariga kaludama"),
    ("Roman Telugu", X, "night ki video call cheyyi, evaru undaru kada"),
    ("Roman Telugu", X, "nee dress lo super ga unnav, inka photos pampu"),
    ("Roman Telugu", T, "nee address naa daggara undi, chudu emavutundo"),
    ("Roman Telugu", T, "inkosari matladite nee sangathi chusukuntanu"),
    ("Roman Telugu", H, "aa kulam vallu andaru dongalu, vallani nammaku"),
    ("Roman Telugu", H, "aadavallaki chaduvu avasaram ledu, intlone undali"),

    # ---------------- Tamil (native script) ----------------
    ("Tamil", S, "இன்று வானிலை நன்றாக இருக்கிறது, வெளியே போகலாம்"),
    ("Tamil", S, "உங்கள் சமையல் குறிப்பு மிகவும் நன்றாக இருந்தது, நன்றி"),
    ("Tamil", O, "நீ ஒரு உபயோகமற்ற முட்டாள், உன் பேச்சை யாரும் கேட்க மாட்டார்கள்"),
    ("Tamil", O, "வாயை மூடு, உன்னால் ஒன்றும் ஆகாது"),
    ("Tamil", X, "உன் புகைப்படத்தை அனுப்பு, யாரிடமும் சொல்ல மாட்டேன்"),
    ("Tamil", X, "நீ ரொம்ப அழகா இருக்க, ஒரு தடவை தனியா சந்திக்கலாமா"),
    ("Tamil", X, "இரவில் வீடியோ கால் பண்ணு, யாரும் இருக்க மாட்டாங்க"),
    ("Tamil", T, "உன் முகவரி என்னிடம் இருக்கு, என்ன நடக்குதுன்னு பாரு"),
    ("Tamil", T, "இன்னொரு தடவை பேசினா உன்னை விட மாட்டேன்"),
    ("Tamil", H, "அந்த சாதிக்காரங்க எல்லாரும் திருடங்க, அவங்கள நம்பாதீங்க"),
    ("Tamil", H, "பெண்களுக்கு படிப்பு தேவையில்லை, அவங்க வீட்டுலயே இருக்கணும்"),

    # ---------------- Tanglish ----------------
    ("Tanglish", S, "indha padam romba nalla irundhuchu, marubadiyum paakkanum"),
    ("Tanglish", S, "naalaikku meeting ethana mani nu sollunga"),
    ("Tanglish", O, "nee oru waste fellow, un pechu yaarum kekka maatanga"),
    ("Tanglish", O, "vaaya moodu da, unnala onnum aagadhu"),
    ("Tanglish", X, "un photo anupu, yaarukkum sollamaaten"),
    ("Tanglish", X, "romba azhaga irukka, oru thadava thaniya sandhikalaama"),
    ("Tanglish", X, "night ku video call pannu, yaarum irukka maatanga"),
    ("Tanglish", X, "un dress la super ah irukka, innum photos anupu"),
    ("Tanglish", T, "un address en kitta iruku, enna nadakkudhu nu paaru"),
    ("Tanglish", T, "innoru thadava pesina un kadhaya mudichiduven"),
    ("Tanglish", H, "andha jaathi kaaranga ellarum thirudanga, avangala nambadheenga"),
    ("Tanglish", H, "pombalainga ku padippu thevai illa, veetlaye irukkanum"),

    # ---------------- Kannada (native script) ----------------
    ("Kannada", S, "ಇವತ್ತು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ, ಹೊರಗೆ ಹೋಗೋಣ"),
    ("Kannada", S, "ನಿಮ್ಮ ಅಡುಗೆ ರೆಸಿಪಿ ತುಂಬಾ ಚೆನ್ನಾಗಿತ್ತು, ಧನ್ಯವಾದಗಳು"),
    ("Kannada", O, "ನೀನು ಒಬ್ಬ ನಿಷ್ಪ್ರಯೋಜಕ ಮೂರ್ಖ, ನಿನ್ನ ಮಾತು ಯಾರೂ ಕೇಳಲ್ಲ"),
    ("Kannada", O, "ಬಾಯಿ ಮುಚ್ಚು, ನಿನ್ನಿಂದ ಏನೂ ಆಗಲ್ಲ"),
    ("Kannada", X, "ನಿನ್ನ ಫೋಟೋ ಕಳಿಸು, ಯಾರಿಗೂ ಹೇಳಲ್ಲ"),
    ("Kannada", X, "ತುಂಬಾ ಚೆನ್ನಾಗಿದ್ದೀಯಾ, ಒಂದು ಸಲ ಒಬ್ಬರೇ ಸಿಗೋಣವಾ"),
    ("Kannada", X, "ರಾತ್ರಿ ವಿಡಿಯೋ ಕಾಲ್ ಮಾಡು, ಯಾರೂ ಇರಲ್ಲ"),
    ("Kannada", T, "ನಿನ್ನ ವಿಳಾಸ ನನ್ನ ಹತ್ರ ಇದೆ, ಏನಾಗುತ್ತೆ ನೋಡು"),
    ("Kannada", T, "ಇನ್ನೊಂದ್ಸಲ ಮಾತಾಡಿದ್ರೆ ನಿನ್ನ ಬಿಡಲ್ಲ"),
    ("Kannada", H, "ಆ ಜಾತಿಯವರೆಲ್ಲ ಕಳ್ಳರು, ಅವರನ್ನ ನಂಬಬೇಡಿ"),
    ("Kannada", H, "ಹೆಂಗಸರಿಗೆ ಓದು ಬೇಡ, ಅವರು ಮನೇಲೇ ಇರಬೇಕು"),

    # ---------------- Kanglish ----------------
    ("Kanglish", S, "ee cinema tumba chennagide, matte nodbeku"),
    ("Kanglish", S, "naale meeting eshtu hottige antha heli"),
    ("Kanglish", O, "neenu ondu waste fellow, ninna maatu yaaru keluvudilla"),
    ("Kanglish", O, "baayi muchhu, ninninda enu aagalla"),
    ("Kanglish", X, "ninna photo kalsu, yaarigu heluvudilla"),
    ("Kanglish", X, "tumba chennagiddiya, ondsala obbare sigona"),
    ("Kanglish", X, "ratri video call maadu, yaaru irolla"),
    ("Kanglish", X, "ninna dress alli super agiddiya, innu photos kalsu"),
    ("Kanglish", T, "ninna address nan hatra ide, enaagutte nodu"),
    ("Kanglish", T, "innondsala maatadidre ninna bidalla"),
    ("Kanglish", H, "aa jaati avarella kallaru, avarnna nambabedi"),
    ("Kanglish", H, "hengasarige odu beda, avaru maneline irbeku"),
]

LANGS = ["English", "Telugu", "Roman Telugu", "Tamil", "Tanglish", "Kannada", "Kanglish"]
CATS = [S, O, X, T, H]
OUT = "saved_models/reports/lang_matrix_80_results.csv"


def main():
    assert len(DATA) == 80, f"expected 80 items, got {len(DATA)}"

    # Local model only - no LLM escalation. HF credits are spent, and the point
    # here is to measure the classifier itself, not the fallback on top of it.
    p = WomenSafetyPredictor(escalate_low_confidence=False)

    texts = [t for _, _, t in DATA]
    preds = p.predict_batch(texts)

    rows = []
    for (lang, expected, text), r in zip(DATA, preds):
        rows.append({
            "language": lang, "expected": expected, "predicted": r.label,
            "correct": r.label == expected, "confidence": round(r.confidence, 3),
            "p_expected": round(r.probabilities[expected], 3),
            "would_escalate": r.confidence < 0.85, "text": text,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    def pct(x):
        return "%5.1f%%" % (100 * x) if x == x else "    -"

    print("\n" + "=" * 74)
    print("OVERALL: %d/%d correct (%.1f%%)" % (df.correct.sum(), len(df), 100 * df.correct.mean()))
    print("=" * 74)

    print("\n--- Accuracy by CATEGORY ---")
    for c in CATS:
        s = df[df.expected == c]
        print("  %-18s %2d/%2d  %s" % (c, s.correct.sum(), len(s), pct(s.correct.mean())))

    print("\n--- Accuracy by LANGUAGE ---")
    for l in LANGS:
        s = df[df.language == l]
        print("  %-14s %2d/%2d  %s" % (l, s.correct.sum(), len(s), pct(s.correct.mean())))

    print("\n--- LANGUAGE x CATEGORY (correct / total) ---")
    print("  %-14s %s" % ("", "  ".join("%-8s" % c[:8] for c in CATS)))
    for l in LANGS:
        cells = []
        for c in CATS:
            s = df[(df.language == l) & (df.expected == c)]
            cells.append("%-8s" % ("%d/%d" % (s.correct.sum(), len(s)) if len(s) else "-"))
        print("  %-14s %s" % (l, "  ".join(cells)))

    print("\n--- CONFUSION (expected -> predicted) ---")
    print(pd.crosstab(df.expected, df.predicted).reindex(index=CATS).fillna(0).astype(int).to_string())

    print("\n--- MISSES that would NOT escalate (conf >= 0.85: silently wrong) ---")
    bad = df[(~df.correct) & (~df.would_escalate)].sort_values("confidence", ascending=False)
    if bad.empty:
        print("  none")
    for _, r in bad.iterrows():
        print("  [%-13s] %-18s -> %-18s conf=%.3f" % (r.language, r.expected, r.predicted, r.confidence))
        print("      %s" % r.text[:88])

    esc = df[~df.correct & df.would_escalate]
    print("\n  (%d further misses fall below 0.85 and WOULD escalate to the LLM)" % len(esc))
    print("\nSaved -> %s" % OUT)


if __name__ == "__main__":
    main()
