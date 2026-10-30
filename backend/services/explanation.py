def explain(a,b,label,probabilities,reasons):
    text = f'{label}: "{a["text"]}" compared with "{b["text"]}". '
    if reasons:
        return text + " ".join(reasons) + " Human review is required."
    if label == "Contradiction":
        return text + "The NLI model judges these claims incompatible under a shared scope. Verify entity, effective period and conditions in the cited passages."
    if label == "Agreement":
        return text + "The model finds supporting meaning in the evaluated direction(s); this is not proof that every detail is identical."
    if label == "Neutral":
        return text + "The model does not establish either contradiction or entailment. Missing information is not itself a conflict."
    return text + "Model confidence or directional consistency is insufficient for an automatic decision."
