import chromadb
import os

client = chromadb.PersistentClient(path=os.path.abspath("./data/chromalocal"))


def get_chroma_client():
    return client
