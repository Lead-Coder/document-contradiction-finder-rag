from functools import lru_cache
from backend.core.config import MODELS, NLI_ID, OFFLINE, DEVICE

def label_mapping(config):
    mapping = {int(i): str(label).lower() for i, label in config.id2label.items()}
    required = {"contradiction", "entailment", "neutral"}
    if set(mapping.values()) != required or set(mapping)!={0,1,2}:
        raise ValueError(f"Model {NLI_ID} has ambiguous id2label {mapping}; configure a three-label NLI checkpoint with semantic labels.")
    inverse=getattr(config,"label2id",None)
    if inverse and {str(k).lower():int(v) for k,v in inverse.items()}!={v:k for k,v in mapping.items()}:
        raise ValueError("Model label2id contradicts id2label")
    return mapping

@lru_cache(maxsize=1)
def load():
    import torch
    import os
    torch.set_num_threads(max(1, int(os.getenv("DOCU_TORCH_THREADS", "4"))))
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(NLI_ID, cache_dir=str(MODELS), local_files_only=OFFLINE)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_ID,
        cache_dir=str(MODELS), local_files_only=OFFLINE).to(DEVICE).eval()
    mapping=label_mapping(model.config)
    import logging
    logging.getLogger(__name__).info("NLI id2label=%s label2id=%s",model.config.id2label,model.config.label2id)
    return tokenizer, model, mapping

def predict(pairs, progress=None, batch_size=8):
    import torch
    if not pairs:return []
    tokenizer, model, mapping = load()
    predictions = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start+batch_size]
        inputs = tokenizer([a for a,b in batch], [b for a,b in batch],
            padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            rows = torch.softmax(model(**inputs).logits, dim=-1).cpu().tolist()
        predictions.extend({mapping[i]: float(value) for i,value in enumerate(row)} for row in rows)
        if progress:
            progress(min(1, (start+len(batch))/len(pairs)))
    return predictions
