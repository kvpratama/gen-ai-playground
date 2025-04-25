import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_cohere import CohereEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class RAG:
    def __init__(self, url):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        llm_model = "gemini-2.0-flash-lite" # "gemma-3-27b-it" # 

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=0.9,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        # self.embeddings = CohereEmbeddings(model="embed-english-v3.0",)
        self.vector_store = InMemoryVectorStore(self.embeddings)

        self.load_and_chunk_url(url)

        # Define prompt for question-answering
        self.prompt = hub.pull("rlm/rag-prompt")

        # Build state graph for the RAG process
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    def load_and_chunk_url(self, url):
        # Load and chunk contents of the website
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    # class_=("post-content", "post-title", "post-header")
                    "main",
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        _ = self.vector_store.add_documents(documents=all_splits)

    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def query(self, question):
        response = self.graph.invoke({"question": question})
        return response["answer"]