def explain(a,b,label,probabilities,reasons,fa=None,fb=None):
    from backend.services.context_verification import features
    fa,fb=fa or features(a),fb or features(b)
    topic=fa["topic"] if fa["topic"]==fb["topic"] else f"{fa['topic']} versus {fb['topic']}"
    evidence=f'Document A states: "{a["text"]}" Document B states: "{b["text"]}". '
    if label=="Neutral":
        return evidence+f"No contradiction is established. Topic comparison: {topic}. "+" ".join(reasons)
    if label=="Policy Change":
        return evidence+f"These provisions differ on {topic}. "+" ".join(reasons)+" Treat this as a policy revision or version difference, not an unconditional simultaneous contradiction."
    if label=="Needs Review":
        return evidence+"No automatic contradiction decision is justified. "+(" ".join(reasons) if reasons else "The NLI signals are weak, directional, or inconsistent; verify the shared scope.")
    if label=="Agreement":
        return evidence+f"The NLI comparison supports agreement on {topic}. "+(" ".join(reasons)+" The cited provision appears unchanged." if reasons else "This does not establish that the entire documents agree.")
    category=fa.get("account_category") or "the stated customer scope"
    return evidence+f"The claims concern {topic} for {category}; no explicit scope mismatch was found. The NLI model indicates incompatible assertions. If their unspecified conditions and effective periods also match, they are a potential conflict. Review the quoted values, permissions and source context."
