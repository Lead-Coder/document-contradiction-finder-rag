"""Conservative, interpretable policy scope checks; no model scores are invented."""
import json
import re
from functools import lru_cache

TOPICS = {
    "withdrawal limit": (r"withdraw|cash.*ceiling", "Daily cash withdrawal maximum amount"),
    "transaction limit": (r"(?:transfer|transaction|payment).*(?:limit|ceiling|maximum|up to)|(?:limit|ceiling|maximum).*transfer", "Maximum money transfer transaction limit"),
    "account maintenance fee": (r"maintenan|account.*(?:monthly|annual|yearly).*(?:fee|charge)", "Monthly annual account maintenance fee charge"),
    "refund processing": (r"refund", "Refund processing completion time business days"),
    "interest rate": (r"interest|\bapr\b", "Annual deposit loan interest percentage rate"),
    "account eligibility": (r"eligible|eligibility|\bminors?\b|minimum age", "Customer eligibility to open an account"),
    "KYC requirements": (r"\bkyc\b|identity|two.factor|authentication", "Mandatory identity verification and authentication requirements"),
    "complaint resolution": (r"complaint", "Customer complaint resolution deadline"),
    "customer support hours": (r"(?:support|help desk|live chat).*(?:hour|weekday|weekend|operat|available|open|monday|friday)", "Customer support opening working hours availability"),
    "account closure": (r"clos(?:e|ing|ure).*account|account.*clos", "Account closure fees and conditions"),
    "payment processing": (r"payment.*process|settlement", "Payment processing and settlement time"),
    "late payment penalty": (r"late.*(?:payment|fee|penalt)|overdue", "Late payment penalty fee"),
    "loan repayment": (r"loan|repayment|installment|instalment", "Loan repayment schedule"),
    "card restrictions": (r"(?:card|international).*(?:permitted|prohibited|allowed|payment|restrict)|(?:permitted|prohibited).*card", "Card payment permissions and restrictions"),
    "failed transaction reversal": (r"revers|failed.*transaction", "Failed transaction automatic reversal timeline"),
    "statement delivery": (r"statement.*(?:deliver|issued|electronic|post)|(?:monthly|account).*statement", "Account statement delivery schedule"),
    "dispute window": (r"dispute", "Card transaction dispute filing deadline"),
    "card replacement": (r"replacement|replace.*card", "Debit card replacement charge"),
    "minimum balance": (r"minimum balance", "Minimum account balance requirement"),
    "account inactivity": (r"inactive|inactivity", "Account inactivity definition duration"),
    "transaction alerts": (r"alert|notification", "Transaction alert notification delivery"),
    "service availability": (r"portal|planned maintenance|service.*available", "Customer portal service availability maintenance"),
}
# Ordered specific topics take priority over broad references (e.g. card refund).
PRIORITY = ["refund processing", "withdrawal limit", "transaction limit", "account maintenance fee",
    "interest rate", "complaint resolution", "customer support hours", "account closure",
    "failed transaction reversal", "statement delivery", "dispute window", "card replacement",
    "minimum balance", "account inactivity", "transaction alerts", "account eligibility",
    "KYC requirements", "late payment penalty", "loan repayment", "card restrictions", "payment processing", "service availability"]


def non_policy(text):
    """Only explicit document furniture/disclaimers; not any mention of policy/version."""
    t = re.sub(r"\s+", " ", text).strip()
    patterns = [r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?[.]?$",
        r"^(?:internal\s+)?(?:policy|document)\s+reference\b", r"^document\s+\d+\s+of\s+\d+\b",
        r"^version\s+\d+(?:\.\d+)*\s*$", r"^effective\s+(?:date[: ]+)?\d",
        r"copyright|all rights reserved", r"\bis a fictional (?:organization|organisation|institution|company)\b",
        r"no (?:policy|statement).*describes? a real", r"fictional training (?:document|material)",
        r"^prepared by[: ]", r"^classification:\s*(?:demo|training)",
        r"(?:reference|document) is intended only for testing", r"^document control$",
        r"^scope and interpretation$", r"^[A-Z][A-Z\s]+\s*[|].*Operations$",
        r"^\S+\s*[|].*Customer Finance Operations$", r"fictional training material.*page\s*\d+"]
    return any(re.search(p, t, re.I if p not in patterns[-3:-1] else 0) for p in patterns)


def rule_topic(text, section_title=""):
    for name in PRIORITY:
        if re.search(TOPICS[name][0], text, re.I):
            return name
    if section_title:
        return rule_topic(section_title)
    return "unknown"


@lru_cache(maxsize=1)
def topic_vectors():
    from backend.services.embeddings import encode
    return encode([TOPICS[name][1] for name in PRIORITY])


@lru_cache(maxsize=2048)
def semantic_topic(text):
    from backend.services.embeddings import encode
    import numpy as np
    scores = np.asarray(topic_vectors()) @ np.asarray(encode([text])[0])
    order = scores.argsort()
    # Abstain on weak or competing topic matches; values are similarities, not probabilities.
    if scores[order[-1]] >= .50 and scores[order[-1]] - scores[order[-2]] >= .08:
        return PRIORITY[order[-1]], float(scores[order[-1]])
    return "unknown", float(scores[order[-1]])


NUMBERS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
    "nine":9,"ten":10,"twelve":12,"fourteen":14,"fifteen":15,"twenty":20,"thirty":30}


@lru_cache(maxsize=4096)
def quantities(text):
    # Normalize for comparison only; original text remains intact for evidence.
    normalized = text.lower().replace("\u20b9", " inr ").replace("rupees", "inr")
    for word,value in NUMBERS.items():
        normalized = re.sub(r"\b" + word + r"\b", str(value), normalized)
    found=[]
    pattern=r"(?:(inr|usd|eur|gbp|rs[.]?)\s*)?(\d[\d,]*(?:\.\d+)?)(?:\s*(%|percent|business days?|working days?|calendar days?|days?|hours?|months?|years?))?"
    for match in re.finditer(pattern, normalized):
        currency,value,unit=match.groups()
        value=float(value.replace(",", ""))
        if 1900 <= value <= 2100 and not currency and not unit:
            continue
        unit=(unit or ("money" if currency else "number")).replace("working", "business")
        if unit in ("%","percent"):unit="percent"
        unit=unit.rstrip("s")
        found.append(dict(value=value,currency=(currency or "").upper().replace("RS.","INR").replace("RS","INR"),unit=unit,source=match.group(0)))
    return found


def features(claim):
    text=claim["text"]
    topic=claim.get("topic") or rule_topic(text,claim.get("section_title", ""))
    groups=re.findall(r"\b(basic|standard|premium|silver|gold|business|student|retail|corporate|senior|minor)\b(?=[^.;]{0,35}\b(?:accounts?|customers?|holders?|deposits?)\b)",text,re.I)
    # Prefer an explicit account tier over broad retail/customer descriptors.
    tier=[g.lower() for g in groups if g.lower() not in ("retail",)]
    category=claim.get("account_category") or ",".join(sorted(set(tier or [g.lower() for g in groups])))
    currencies=set(re.findall(r"\b(INR|USD|EUR|GBP)\b",text,re.I))
    if "\u20b9" in text or re.search(r"\brupees?\b",text,re.I):currencies.add("INR")
    currency=",".join(sorted(c.upper() for c in currencies)) or claim.get("currency", "")
    periods=[]
    for name,pattern in [("daily",r"daily|per day|each day"),("monthly",r"monthly|per month|each month"),("annual",r"annual|yearly|per year|each year")]:
        if re.search(pattern,text,re.I):periods.append(name)
    conditions=re.findall(r"\b(?:if|unless|provided that|subject to)\s+([^.;]+)",text,re.I)
    years=sorted(set(re.findall(r"\b(?:19|20)\d{2}\b",text)))
    bound="exact" if re.search(r"\bexactly\b|\b(?:limit|ceiling|rate|fee)\b.*\b(?:is|of)\b",text,re.I) else "unspecified"
    if re.search(r"within|up to|at most|no more than",text,re.I):bound="upper"
    if re.search(r"at least|or (?:later|higher|greater)|minimum",text,re.I):bound="lower"
    if re.search(r"typically|usually|target(?:ed)?|average",text,re.I):bound="typical"
    return dict(topic=topic,account_category=category,currency=currency,period=",".join(periods),
        years=years,conditions=[s.lower().strip() for s in conditions],quantities=quantities(text),bound=bound,
        version=claim.get("version", ""),effective_date=claim.get("effective_date", ""),
        organization=claim.get("organization", ""),geography=claim.get("geography", ""))


def candidate_check(a,b,similarity=1.0,min_similarity=.35,semantic=False):
    if non_policy(a["text"]) or non_policy(b["text"]):
        return False,"Document furniture or disclaimer is not a policy claim.",features(a),features(b)
    fa,fb=features(a),features(b)
    if semantic and ((fa["topic"]=="unknown") != (fb["topic"]=="unknown")):
        if fa["topic"]=="unknown":fa["topic"],_=semantic_topic(a["text"])
        if fb["topic"]=="unknown":fb["topic"],_=semantic_topic(b["text"])
    if similarity < min_similarity:
        return False,f"Retrieval cosine similarity {similarity:.3f} is below {min_similarity:.3f}.",fa,fb
    if fa["topic"]!="unknown" and fb["topic"]!="unknown" and fa["topic"]!=fb["topic"]:
        return False,f"Different policy attributes: {fa['topic']} versus {fb['topic']}.",fa,fb
    if (fa["topic"]=="unknown") != (fb["topic"]=="unknown"):
        return False,"Only one claim has an established financial policy attribute.",fa,fb
    return True,"Compatible retrieval topic; scope still requires verification.",fa,fb


def numeric_interval(text,values):
    """Interpret only explicit one-dimensional bounds; abstain on complex formulae."""
    text=text.lower()
    if len(values)==2 and re.search(r"\bbetween\b",text):
        low,high=values[0]["value"],values[1]["value"]
        return (low,high,True,True) if low<=high else None
    if len(values)!=1:return None
    value=values[0]["value"]
    if re.search(r"at least|minimum|or (?:higher|greater|later)",text):return (value,None,True,False)
    if re.search(r"more than|greater than|strictly above",text):return (value,None,False,False)
    if re.search(r"within|up to|at most|no more than",text):return (None,value,False,True)
    if re.search(r"less than|strictly below",text):return (None,value,False,False)
    if re.search(r"exactly",text):return (value,value,True,True)
    return None


def intervals_overlap(a,b):
    lo=max(float("-inf") if a[0] is None else a[0],float("-inf") if b[0] is None else b[0])
    hi=min(float("inf") if a[1] is None else a[1],float("inf") if b[1] is None else b[1])
    if lo<hi:return True
    if lo>hi:return False
    contains=lambda r:(r[0] is None or lo>r[0] or lo==r[0] and r[2]) and (r[1] is None or lo<r[1] or lo==r[1] and r[3])
    return contains(a) and contains(b)


def scope(a,b,fa=None,fb=None):
    fa,fb=fa or features(a),fb or features(b)
    warnings=[];changes=[]
    for field,title in [("account_category","Account/customer categories"),("currency","Currencies"),
        ("period","Measurement periods"),("organization","Organizations"),("geography","Geographic scopes")]:
        if fa[field] and fb[field] and fa[field].casefold()!=fb[field].casefold():
            warnings.append(f"{title} differ: {fa[field]} versus {fb[field]}.")
        elif bool(fa[field]) != bool(fb[field]) and field in ("account_category","currency","period","organization"):
            warnings.append(f"{title} are specified on only one side; shared scope is uncertain.")
    if fa["conditions"]!=fb["conditions"] and (fa["conditions"] or fb["conditions"]):
        warnings.append("Applicable conditions differ or are omitted on one side.")
    if fa["version"] and fb["version"] and fa["version"]!=fb["version"]:
        changes.append(f"Policy versions differ: {fa['version']} versus {fb['version']}.")
    if fa["effective_date"] and fb["effective_date"] and fa["effective_date"]!=fb["effective_date"]:
        changes.append(f"Effective dates differ: {fa['effective_date']} versus {fb['effective_date']}.")
    if fa["years"] and fb["years"] and set(fa["years"]).isdisjoint(fb["years"]):
        changes.append("Explicit policy years differ.")
    ua={q["unit"] for q in fa["quantities"] if q["unit"]!="number"}
    ub={q["unit"] for q in fb["quantities"] if q["unit"]!="number"}
    if ua and ub and ua.isdisjoint(ub):warnings.append("Measurement units differ; no unit conversion is assumed.")
    qa,qb=fa["quantities"],fb["quantities"]
    different=qa and qb and {(q["value"],q["unit"],q["currency"]) for q in qa}!={(q["value"],q["unit"],q["currency"]) for q in qb}
    if different and fa["bound"]==fb["bound"] and fa["bound"] in ("upper","lower","typical"):
        warnings.append(f"Both values are {fa['bound']} bounds or typical commitments; different values can be compatible.")
    ia,ib=numeric_interval(a["text"],qa),numeric_interval(b["text"],qb)
    if different and ia and ib and intervals_overlap(ia,ib) and not any("bounds" in w for w in warnings):
        warnings.append("The explicit numerical ranges overlap, including their stated endpoints; different constraints need not conflict.")
    if a.get("fragmented") or b.get("fragmented"):warnings.append("A long claim was fragmented; inspect its original parent passage.")
    return dict(warnings=warnings,policy_changes=changes,compatible=not warnings and not changes)


def verify(a,b):
    """Compatibility wrapper retained for existing tests/evaluation callers."""
    checked=scope(a,b)
    reasons=checked["warnings"]+checked["policy_changes"]
    left,right=a.get("parent",a["text"]),b.get("parent",b["text"])
    years=lambda t:set(re.findall(r"\b(?:19|20)\d{2}\b",t))
    if years(left) and years(right) and years(left).isdisjoint(years(right)) and "Explicit policy years differ." not in reasons:
        reasons.append("Explicit policy years differ.")
    dates=lambda t:set(x.lower() for x in re.findall(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?|\b\d{4}-\d{2}-\d{2}\b",t,re.I))
    if dates(left) and dates(right) and dates(left).isdisjoint(dates(right)):reasons.append("Explicit calendar dates differ.")
    entity=lambda t:set(re.findall(r"\b(?:Department|Project|Product|Team|Campus)\s+([A-Z0-9][\w-]*)",t))
    if entity(left) and entity(right) and entity(left).isdisjoint(entity(right)):reasons.append("Named entity identifiers differ.")
    stop={"the","and","for","are","was","were","has","have","had","must","may","can","not","with","from","that","this","their","they","every","within","after","before","there","its","our","will","shall","does","into","than","only","both","contains","provides"}
    terms=lambda t:{w.rstrip("s") for w in re.findall(r"[a-z]{3,}",t.lower()) if w not in stop}
    if features(a)["topic"]==features(b)["topic"]=="unknown" and not terms(a["text"]) & terms(b["text"]):
        reasons.append("No shared content terms establish a common topic or referent.")
    if re.search(r"\bor (?:later|higher|greater)\b",a["text"]) and re.search(r"\bor (?:later|higher|greater)\b",b["text"]) and not reasons:
        reasons.append("Compatible lower bounds do not establish conflict.")
    return reasons
