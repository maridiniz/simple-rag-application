# Importing all necessary dependencies:
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
import os

def main():   
   print("--- Starting RAG application setup ---")

   documents = text_loading_splitting()
   if not documents:
       print("No documents loaded. Exiting.")
       return
   
   vector_store = vectorization(documents)

   llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

   print("--- RAG application setup complete. Building graph --- ")

   while True:
       rag_app = langgraph_wrapper(vector_store, llm)

       question = input("Ask a question: ")

       response = rag_app.invoke({"question": question})

       print({"Answer": response.get("answer", "No answer generated.")})

       user_replay = input("Ask another question? (yes/no)")

       if user_replay not in ["yes"]:
           return
       else:
           continue


# Load documents and split text content in small chunks:
def text_loading_splitting(directory_path: str = "./files") -> List[Document]:
    """Load documents and split text."""

    # Check if the directory exists:
    try:
        if not os.path.exists("./files"):           
            return []
        
        loader = PyPDFDirectoryLoader(path=directory_path, glob="**/*.pdf")  
        docs = loader.load()
        if not docs:
            print("No pdf files were found in the given diectory.")
            return []
    except Exception as e:
        print(f"Error loading documents from: {directory_path}: {e}")
        return
    
    # Process text to split it into small chuncks:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

    all_splits = text_splitter.split_documents(documents=docs)

    return all_splits


# Indexing and storing the splitted text with Chromaclient:
def vectorization(documents: List[Document], persist_directory: str = "../../../chroma_langchain_db"):
    """Index text splitted and store it into a Chroma db."""

    # create an embedding instance:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Trying initializing Chroma client:
    print(f"Trying to initialize Chroma client in {persist_directory}, it may take a moment...\n")

    # If persist directory does not exists, the os.makedirs func ceates it:
    os.makedirs(persist_directory, exist_ok=True) 

    try:     
        vector_store = Chroma(
            collection_name="pdf_files",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        if vector_store._collection.count() == 0 and documents:
            print("Vector store is empty or new. Adding documents...")
            vector_store.add_documents(documents=documents)
            print(f"Added {len(documents)} documents to Chroma.")
        
        elif documents:
            print("Vector stores already exists. Checking for new documents to add.")
            existing_ids = {id.get("id") for id in vector_store.get(include=["metadatas"])["metadatas"]}        

            new_docs_to_add = [doc for doc in documents if doc.metadata.get("id") not in existing_ids]
            if new_docs_to_add:
                print(f"Adding {len(new_docs_to_add)} new documents to existing vector store.")
                vector_store.add_documents(new_docs_to_add)
            else:
                print("No documents to add to existing vector store.")
        else:
            print("Vector store initialized, but no documents provided to addition.")

    except Exception as e:
        print(f"Error initializing or updating Chroma:  {e}")
        
        # Indexing and storing text in Chroma db:
        if documents:
            print("Attempting to create a new Chroma collection as an alternative.")
            vector_store = Chroma.from_documents(
                documents=documents,
                collection_name="pdf_files",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print(f"Successfully created new Chroma collection with {len(documents)} documents ")
        else:
            raise RuntimeError(f"Could not initialize Chroma and no documents to create a new one: {e}")

    
    return vector_store


# Define a class to set the structure of the question, context and answer:
class State(TypedDict):
    """Represents the state of our graph.
    
    Attributes: 
        question: The user question.
        context: List of retrieved documents.
        answer: The generated answer.
    """

    question: str
    context: List[Document]
    answer: str


# Define a function to retrieve:
def retrieve(state: State, vector_store: Chroma) -> dict:
    """Retrieves the information from out index texts."""

    doc_retrieved = vector_store.similarity_search(state["question"])
    return {"context": doc_retrieved}


# Define a function to generate the answer to the user:
def generate(state: State, llm: ChatGoogleGenerativeAI) -> dict:
    """Takes the context, question and generate an answer."""

    # Defining the llm custom template:
    template = """"
    Your are an AI assistent for answering questions based on the provide context.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say you don't know, don't try to make up an  answer.
    Keep the answer concise an to the point.

    Context: {context}

    Question: {question}

    Answer:
    """

    rag_template = ChatPromptTemplate.from_template(template)

    # Set the whole message to the user with the llm:
    doc_content = "\n\n".join(doc.page_content for doc in state["context"])
    message = rag_template.invoke({"question": state["question"], "context": doc_content})
    response = llm.invoke(message)

    return {"answer": response.content}


# Wrap all the application steps with langgraph:
def langgraph_wrapper(vector_store: Chroma, llm: ChatGoogleGenerativeAI):

    workflow = StateGraph(State)
    workflow.add_node("retrieve", lambda state: retrieve(state, vector_store))
    workflow.add_node("generate", lambda state: generate(state, llm))
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()



# Call main function:
if __name__ == "__main__":
    main()