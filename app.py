from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import warnings
warnings.filterwarnings("ignore")

def load_documents():
    document_loader = PyPDFDirectoryLoader("data")
    return document_loader.load()

documents = load_documents()
print(documents[0])