import os
import warnings
import pathlib

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Suppress LangChain deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="langchain"
)

# Ensure your OpenAI API key is set securely
os.environ["OPENAI_API_KEY"] = "sk-OaYAPBBVMBqy9a0vg5zedounXniUFwhwfVN9L6Fa2hT3BlbkFJWw1QtLRmHGYgy5rq-9hUI4BVEvV4vflpSiDbmvHf0A"  # Replace with your actual API key

class PersistentChromaQA:
    def __init__(
        self,
        pdf_dir1,
        pdf_dir2,
        persist_dir="chroma_db",
        chunk_size=1500,
        chunk_overlap=200,
        top_k=5,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    ):
        self.pdf_dir1 = pdf_dir1
        self.pdf_dir2 = pdf_dir2
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)

        if self._is_persist_dir_available():
            print(f"Loading existing Chroma store from '{self.persist_dir}'...")
            self.vectorstore = self._load_chroma_store()
        else:
            print(f"No existing Chroma store found at '{self.persist_dir}'. Creating a new one...")
            self.vectorstore = self._create_chroma_store()

        self.qa_chain = self._build_qa_chain()

    def _is_persist_dir_available(self):
        path = pathlib.Path(self.persist_dir)
        return path.exists() and any(path.iterdir())

    def _load_chroma_store(self):
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )

    def _create_chroma_store(self):
        all_docs = self._load_all_documents()
        if not all_docs:
            raise ValueError("No documents found. Cannot create a Chroma store.")

        splitted_docs = self._chunk_documents(all_docs)

        vectorstore = Chroma.from_documents(
            splitted_docs,
            self.embeddings,
            persist_directory=self.persist_dir
        )
        vectorstore.persist()
        print(f"Chroma store created and persisted at '{self.persist_dir}'.")
        return vectorstore

    def _load_all_documents(self):
        all_docs = []

        # Load PDFs from first directory
        pdf_files1 = [f for f in pathlib.Path(self.pdf_dir1).glob("*.pdf")]
        for pdf_file in pdf_files1:
            loader = PyPDFLoader(str(pdf_file))
            try:
                pdf_docs = loader.load()
                all_docs.extend(pdf_docs)
                print(f"Loaded {len(pdf_docs)} pages from '{pdf_file}'.")
            except Exception as e:
                print(f"Error loading '{pdf_file}': {e}")

        # Load PDFs from second directory
        pdf_files2 = [f for f in pathlib.Path(self.pdf_dir2).glob("*.pdf")]
        for pdf_file in pdf_files2:
            loader = PyPDFLoader(str(pdf_file))
            try:
                pdf_docs = loader.load()
                all_docs.extend(pdf_docs)
                print(f"Loaded {len(pdf_docs)} pages from '{pdf_file}'.")
            except Exception as e:
                print(f"Error loading '{pdf_file}': {e}")

        return all_docs

    def _chunk_documents(self, documents):
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def _build_qa_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=self.llm_model, temperature=self.temperature),
            chain_type="stuff",
            retriever=retriever
        )
        return chain

    def query(self, user_question):
        if not user_question:
            raise ValueError("The user_question cannot be empty.")

        answer = self.qa_chain.run(user_question)
        return answer


# ----------------------
# Example usage
# ----------------------
if __name__ == "__main__":
    pdf_directory1 = "./CFR_33"
    pdf_directory2 = "./CFR_40"

    qa_system = PersistentChromaQA(
        pdf_dir1=pdf_directory1,
        pdf_dir2=pdf_directory2,
        persist_dir="chroma_db",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=15,
        embedding_model="text-embedding-ada-002",
        llm_model="gpt-3.5-turbo",
        temperature=0.0
    )

    user_question = "hey, so my facility just had an oil spill of 500 gallons.  what do I need to do?"
    answer = qa_system.query(user_question)

    print("\nAnswer to your question:")
    print(answer)
