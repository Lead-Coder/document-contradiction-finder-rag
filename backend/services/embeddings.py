from functools import lru_cache
from backend.core.config import MODELS, EMBED_ID, OFFLINE, DEVICE

@lru_cache(maxsize=1)
def model():
    import os
    import torch
    torch.set_num_threads(max(1, int(os.getenv("DOCU_TORCH_THREADS", "4"))))
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_ID, cache_folder=str(MODELS),
        local_files_only=OFFLINE, device=DEVICE)

def encode(texts):
    return model().encode(texts, batch_size=32, normalize_embeddings=True,
        show_progress_bar=False).tolist()
