# Importando todas as dependências necessárias:
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
import os

# Corpo principal da nossa aplicação:
def main():   
   print("--- Iniciando Aplicação ---")

   documents = text_loading_splitting()
   if not documents:
       print("Nenhum documento carregado. Saindo.")
       return
   
   vector_store = vectorization(documents)

   llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

   print("--- Bem-vindo a aplicação RAG --- ")

   while True:
       rag_app = langgraph_wrapper(vector_store, llm)

       question = input("Faça uma pergunta: ")

       response = rag_app.invoke({"question": question})

       print({"Resposta": response.get("answer", "Nenhuma resposta gerada.")})

       user_replay = input("Fazer uma nova pergunta? (sim/não)")

       if user_replay not in ["sim"]:
           return
       else:
           continue


# 1. Passo, define o carregamento e divisão dos arquivos em partes menores:
def text_loading_splitting(directory_path: str = "../docs") -> List[Document]:
    """Carrega os documentos e divide em várias partes menores ."""

    # Lida com possíveis erros em diretórios ou arquivos inexistêntes:
    try:
        if not os.path.exists("../docs"):           
            return []
        
        loader = PyPDFDirectoryLoader(path=directory_path, glob="**/*.pdf")  
        docs = loader.load()
        if not docs:
            print("Nenhum arquivo PDF encontrado no diretório.")
            return []
    except Exception as e:
        print(f"Erro ao carregar documentos de: {directory_path}: {e}")
        return
    
    # Processa os arquivos e divide em partes menores:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

    all_splits = text_splitter.split_documents(documents=docs)

    return all_splits


# 2. Passo, responsável pela vetorização e indexação dos arquivos já processados:
def vectorization(documents: List[Document], persist_directory: str = "../vector_store"):
    """vetoriza e indexa os arquivos em um diretório local."""

    # create an embedding instance:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Trying initializing Chroma client:
    print(f"Tentando iniciar o cliente Chroma em: {persist_directory}, pode levar um momento...\n")

    # Verifica o diretório, caso não exista, makedirs cria o diretório:
    os.makedirs(persist_directory, exist_ok=True) 

    # Tenta iniciar Chroma no local indicado em persist_directory:
    try:     
        vector_store = Chroma(
            collection_name="pdf_files",
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        if vector_store._collection.count() == 0 and documents:
            print("vector store está vazio ou é novo. Adicionando documentos...")
            vector_store.add_documents(documents=documents)
            print(f"Adicionado {len(documents)} documentos no Chroma.")
        
        elif documents:
            print("vector stores já existe. Checando novos documentos para adicionar.")
            existing_ids = {id.get("id") for id in vector_store.get(include=["metadatas"])["metadatas"]}        

            new_docs_to_add = [doc for doc in documents if doc.metadata.get("id") not in existing_ids]
            if new_docs_to_add:
                print(f"Adicionando {len(new_docs_to_add)} novos documentos no vector store existente.")
                vector_store.add_documents(new_docs_to_add)
            else:
                print("Sem documentos para adicionar no vector store existente.")
        else:
            print("Vector store inicializado, mas sem documentos para adicionar.")

    except Exception as e:
        print(f"Erro ao inicializar ou atualizar Chroma:  {e}")
        
        # Indexing and storing text in Chroma db:
        if documents:
            print("Tentativa de criar uma nova coleção Chroma como alternativa.")
            vector_store = Chroma.from_documents(
                documents=documents,
                collection_name="pdf_files",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            print(f"Sucesso em criar uma nova coleção com {len(documents)} documentos.")
        else:
            raise RuntimeError(f"Não foi possível iniciar Chroma: {e}")

    
    return vector_store


# Classe para definir o estado da nossa aplicação:
class State(TypedDict):

    question: str
    context: List[Document]
    answer: str


# 3. Passo, define a busca dos arquivos indexados:
def retrieve(state: State, vector_store: Chroma) -> dict:
    """Realiza busca nos arquivos indexados."""

    doc_retrieved = vector_store.similarity_search(state["question"])
    return {"context": doc_retrieved}


# 4. Passo, gera uma resposta ao usuário com o llm:
def generate(state: State, llm: ChatGoogleGenerativeAI) -> dict:
    """Pega os dados buscados, a pergunta e gera uma resposta com o llm."""

    # Define o template customizado para o llm:
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

    # Configura todo o processo da mensagem que será passada ao llm:
    doc_content = "\n\n".join(doc.page_content for doc in state["context"])
    message = rag_template.invoke({"question": state["question"], "context": doc_content})
    response = llm.invoke(message)

    return {"answer": response.content}


# 5. passo, agrupa todos os passos da aplicação com LangGraph:
def langgraph_wrapper(vector_store: Chroma, llm: ChatGoogleGenerativeAI):

    workflow = StateGraph(State)
    workflow.add_node("retrieve", lambda state: retrieve(state, vector_store))
    workflow.add_node("generate", lambda state: generate(state, llm))
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()



# Chama a função principal main:
if __name__ == "__main__":
    main()
