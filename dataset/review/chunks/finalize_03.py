# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('chunk_03_ruled.csv')

# ---- fix false-positive misogyny marker: remove overly broad trigger, keep only explicit one ----
def refine(row):
    text = str(row['comment'])
    if 'women ki respect ledhu' in text.lower() and row['rule_verdict']=='Hate Speech':
        return 'Offensive'
    return row['rule_verdict']
df['rule_verdict'] = df.apply(refine, axis=1)

# ---- manual overrides identified via close reading / spot-check ----
overrides = {
    'C017359': ('Safe', 'Caste-identity statement, not hateful'),
    'C017669': ('Safe', 'Friendly compliment, no hate/offense'),
    'C017958': ('Offensive', 'Generic dismissive insult, no group target'),
    'C018346': ('Safe', 'Movie character reference, neutral'),
    'C018561': ('Safe', 'Compliment about a child, neutral'),
    'C018710': ('Safe', 'Anti-caste-discrimination opinion, not hateful'),
    'C018891': ('Threat', 'Explicit threat of violence'),
    'C019005': ('Safe', 'Benign remark about marriage/family'),
    'C019034': ('Unsure', 'Political opinion vs communal framing, ambiguous'),
    'C019351': ('Offensive', 'Mocking hypocrisy, mild insult, no group hate'),
    'C019385': ('Safe', 'Anti-caste opinion, not hateful'),
    'C019916': ('Threat', 'Explicit call to violence'),
    'C019966': ('Unsure', 'Garbled text, meaning unclear'),
    'C020107': ('Unsure', 'Garbled text, meaning unclear'),
    'C020291': ('Safe', 'Anti-caste equality opinion, not hateful'),
    'C020591': ('Unsure', 'Garbled text, meaning unclear'),
    'C021147': ('Unsure', 'Garbled text, meaning unclear'),
    'C021185': ('Hate Speech', 'Communal slur, calls group dogs/traitors'),
    'C021395': ('Safe', 'Devotional political praise, not hateful'),
    'C021444': ('Threat', 'Threat of violence mixed with caste talk'),
    'C021467': ('Safe', 'Benign TV/fan comment'),
    'C021561': ('Safe', 'Benign fan-rivalry speculation'),
    'C021598': ('Safe', 'Self-deprecating anti-caste remark'),
    'C021669': ('Safe', 'Anti-caste film criticism, not hateful'),
    'C021819': ('Hate Speech', 'Anti-Christian slur, incites communal conflict'),
    'C016790': ('Offensive', 'Vulgar personal insult, not group hate'),
    'C021746': ('Offensive', 'Fan-war trash talk, no slur'),
    'C017061': ('Unsure', 'Fragment too short/unclear to judge'),
    'C018263': ('Unsure', 'Sarcastic political jab, ambiguous intent'),
    'C017587': ('Unsure', 'Real-crime caste framing, ambiguous intent'),
    'C020573': ('Offensive', 'Vulgar mockery of hypocrisy, not hate itself'),
    'C016629': ('Safe', 'Mild political criticism, no slur'),
    'C018247': ('Sexual Harassment', 'Crude sexual remark about a named woman'),
    'C021103': ('Safe', 'Criticism of media bias, not hateful'),
    'C021146': ('Offensive', 'Serious personal insult/accusation, not group hate'),
    'C016592': ('Offensive', 'Vulgar personal insult, not group-targeted hate'),
    'C018121': ('Hate Speech', 'Caste-based dehumanizing slur, calls group dogs'),
    'C018604': ('Hate Speech', 'Communal rhetoric otherizing group over food practice'),
    'C018625': ('Hate Speech', 'Dehumanizes group as dogs, love-jihad narrative'),
    'C018917': ('Hate Speech', 'Communal Hindu-Muslim confrontational hate speech'),
    'C021519': ('Hate Speech', 'Dehumanizing caste-group slur "dogs" plus insult'),
    'C021602': ('Hate Speech', 'Blames a caste/lineage group for historic assault'),
    'C021646': ('Hate Speech', 'Caste-based dehumanizing insult, "caste trash"'),
    'C021682': ('Hate Speech', 'Caste-pride rhetoric with implicit communal threat'),
}

def apply_override(row):
    cid = row['comment_id']
    if cid in overrides:
        return overrides[cid][0]
    return row['rule_verdict']

def override_reason(row):
    cid = row['comment_id']
    if cid in overrides:
        return overrides[cid][1]
    return None

df['final_category'] = df.apply(apply_override, axis=1)
df['override_reason'] = df.apply(override_reason, axis=1)

def default_reason(row):
    fc = row['final_category']
    orig = row['category']
    if fc == orig:
        if fc == 'Hate Speech':
            return 'Explicit group-based hate / slur, correctly flagged'
        else:
            return 'Genuine rude/vulgar insult, correctly flagged'
    if fc == 'Safe':
        return 'Neutral movie/political comment, no hate or offense'
    if fc == 'Offensive':
        return 'Rude/vulgar insult but no protected-group target'
    if fc == 'Hate Speech':
        return 'Explicit caste/religion/gender-based hate speech'
    if fc == 'Threat':
        return 'Explicit threat of violence'
    if fc == 'Sexual Harassment':
        return 'Explicit sexual remark targeting a woman'
    return ''

def reason(row):
    if pd.notna(row['override_reason']):
        return row['override_reason']
    return default_reason(row)

df['reason'] = df.apply(reason, axis=1)

def verdict(row):
    if row['final_category'] == 'Unsure':
        return 'Unsure'
    if row['final_category'] == row['category']:
        return 'Correct'
    return 'Wrong'

df['verdict'] = df.apply(verdict, axis=1)

def corrected(row):
    if row['verdict'] == 'Wrong':
        return row['final_category']
    return ''

df['corrected_category'] = df.apply(corrected, axis=1)

out = df[['comment_id','verdict','corrected_category','reason']].copy()
print(out['verdict'].value_counts())
print()
print(out.head(20).to_string())

out.to_csv('verdict_03.csv', index=False, encoding='utf-8-sig')
print('\nWrote verdict_03.csv rows:', len(out))
