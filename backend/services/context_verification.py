import re

def verify(a,b):
    reasons=[]
    # Narrow, explicit scope checks abstain rather than asserting that two
    # statements can both be true. These heuristics do not implement full NER.
    def context(s):
        parent=s.get("parent",s["text"])
        return parent if len(parent)<1800 else s["text"]
    left,right=context(a),context(b)
    # An NLI checkpoint trained on paired scenes can treat unrelated subjects
    # as mutually exclusive. Lexical support is a conservative scope guard,
    # never evidence of contradiction. Zero-overlap paraphrases may abstain.
    stop={"the","and","for","are","was","were","has","have","had","must","may","can",
        "not","with","from","that","this","their","they","every","within","after","before",
        "there","its","our","will","shall","does","into","than","only","both","contains","provides"}
    def terms(text):
        return {word.rstrip("s") for word in re.findall(r"[a-z]{3,}",text.lower()) if word not in stop}
    if not (terms(a["text"]) & terms(b["text"])):
        reasons.append("No shared content terms establish a common topic or referent; semantic paraphrases require human review.")
    lower_bound=r"\b\d+(?:\.\d+)*\s+or\s+(?:later|higher|greater|more)\b|\bat least\s+\d+"
    if re.search(lower_bound,a["text"],re.I) and re.search(lower_bound,b["text"],re.I):
        reasons.append("Both claims specify lower bounds; a stricter minimum may satisfy both, so numerical differences are insufficient to establish conflict.")
    years=lambda t:set(re.findall(r"\b(?:19|20)\d{2}\b",t))
    if years(left) and years(right) and years(left).isdisjoint(years(right)):
        reasons.append("Explicit years differ; the policy periods may differ.")
    dates=lambda t:set(x.lower() for x in re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?|\b\d{4}-\d{2}-\d{2}\b",t,re.I))
    if dates(left) and dates(right) and dates(left).isdisjoint(dates(right)):
        reasons.append("Explicit calendar dates differ; inspect effective dates and event identity.")
    entities=lambda t:set(x.lower() for x in re.findall(r"\b(?:Department|Project|Product|Team|Campus)\s+([A-Z0-9][\w-]*)",t))
    if entities(left) and entities(right) and entities(left).isdisjoint(entities(right)):
        reasons.append("Named entity identifiers differ.")
    conditions=lambda t:set(x.lower().strip(' .') for x in re.findall(r"\b(?:if|unless|provided that|when)\s+([^.;]+)",t,re.I))
    if conditions(left)!=conditions(right) and (conditions(left) or conditions(right)):
        reasons.append("Conditions differ or one statement omits a condition.")
    if a.get("fragmented") or b.get("fragmented"):
        reasons.append("A long claim was split; inspect the parent passage.")
    return reasons
