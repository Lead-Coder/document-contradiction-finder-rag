import os
os.environ["DOCU_OFFLINE"]="false"
from backend.services.embeddings import model
from backend.services.nli_classifier import load

def main():
    print("Downloading embedding model...")
    model()
    print("Downloading NLI model and validating semantic label mapping...")
    _,nli,mapping=load()
    print("NLI labels:",mapping)
    print("Models cached. Runtime defaults to offline.")

if __name__=="__main__":
    main()
