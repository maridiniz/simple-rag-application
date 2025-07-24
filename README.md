# Aplicação RAG Simples
[![LangChain](https://img.shields.io/badge/LangChain-4CAF50?style=flat-square&logo=langchain&logoColor=black)](https://python.langchain.com)
[![PyPI version](https://badge.fury.io/py/langchain.svg)](https://pypi.org/project/langchain/)
[![Downloads](https://static.pepy.tech/badge/langchain)](https://pepy.tech/project/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-9E9E9E?style=flat-square&logo=python&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![PyPI version](https://badge.fury.io/py/langgraph.svg)](https://pypi.org/project/langgraph/)
[![Downloads](https://static.pepy.tech/badge/langgraph)](https://pepy.tech/project/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Objetivo do projeto
O projeto tem o intuito apenas de demonstrar o passo a passo de uma simples aplicação RAG (Retrieval-Augmented Generation). São utilizados os components dos frameworks LangChain, langGraph e alguns components integrados como Google GenAI para os passos de vetorização e geração do output, assim como também o Chroma client para indexação dos embeddings localmente (no caso deste projeto) e posteriormente o passo de recuperação. Também é possível utilizar ainda em conjunto outros frameworks para observabilidade da aplicação, como por example o LangSmith, onde é possível monitorar, realizar debbuging e etc. Existem alguns exemplos de como configurar essa etapa, caso tenha intersse, veja a [documentação oficial](https://docs.smith.langchain.com/observability). É possível encontrar todas as informações diretamenta na documentação official do LangChain neste [link](https://python.langchain.com/docs/introduction/). Quanto a documentação official do LangGraph, acesse [aqui.](https://langchain-ai.github.io/langgraph/)

O RAG é uma aplicação no modelo de perguntas e respostas, onde é realizada uma busca nos nossos próprios arquivos, ao final, tanto a pergunta quanto o conteúdo recuperado (resposta) são passados ao modelo de LLM escolhido para gerar um output mais amigável ao usuário. 


# Setup
Para este projeto serão necessárias as sequintes dependências:
- `langchain-google-genai` ou `langchain_openai` entre outros [modelos](https://python.langchain.com/docs/integrations/text_embedding/) que desejar.
- `langchain-chroma` dentre vários outros clientes [disponíveis](https://python.langchain.com/docs/integrations/vectorstores/).
- `langchain-community >= 0.3.27`
- `langchain-text-splitters >= 0.3.8`
- `langchain-core >= 0.3.68`
- `langchain >= 0.3.26`

Dependências nativas:
- `os`
- `typing_extensions`

Para  realizar a instalação das dependências citadas acima, execute no terminal o comando:
```python
pip install langchain-google-genai langchain-chroma langchain-community langchain-text-splitters langchain-core langchain
```

Para obter uma cópia deste repositório, execute o comando:
```bash
git clone git@github.com:maridiniz/simple-rag-application.git
```


Para prosseguir com o projeto é necessário utilizar um LLM que fará o processo de vetorização, indexação e também participará do processo de gerar o output para o usuário, por tanto,  será necessária utilizar uma chave de API de algum modelo de LLM ou utilizar um modelo local. Para definir a chave de API do modelo LLM escolhido em uma variável de ambiente:

1. opção através do powersehll (Método persiste mesmo após reiniciar o sistema):
```powershell
[System.Environment]::SetEnvironmentVariable('GOOGLE_API_KEY', 'insira aqui sua chave', 'User')
```

2. opção ataravés do CMD (Método permance mesmo após reiniciar):
```cmd
setx GOOGLE_API_KEY "insira-aqui-sua-chave"
```

Para verificar se funcionou:
```powershell
echo $env:GOOGLE_API_KEY
```

```cmd
echo %GOOGLE_API_KEY%
```

Outra alternativa com o módulo getpass:
```python
import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Insira sua chave de API aqui: ")
```

Para observabilidade da aplicação, é possível integrar ao `LangSmith`, um framework que tem todas as funcionalidades de debbuging, monitoração etc. Comece instalando as dependências:
```python
pip install -U langsmith python-dotenv
```

Em seguida, temos algumas opções de como estabelecer a observabilidade com LangSmith, na própria [página oficial](https://docs.smith.langchain.com/observability) também tem as informações:

```python
# 1. Opção é criar um arquivo com a extensão .env:
1 LANGSMITH_TRACING="true"
2 LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
3 LANGSMITH_API_KEY="<your-api-key>"
4 LANGSMITH_PROJECT="pr-crazy-formula-87"
5 OPENAI_API_KEY="<your-openai-api-key>"
```

Esse arquivo pode ser colocado no mesmo diretório do script da aplicação, caso haja apenas um script, mas se houver múltiplos scripts, pode ser colocado no diretório root, ou mesmo specificar o diretório na função `load_dotenv()`, [aqui](https://pypi.org/project/python-dotenv/) tem as informações oficiais do módulo `dotenv` e os componentes.

```python
# Impotando dependência:
from dotenv import load_dotenv

# A função carrega as variáveis do arquivo `.env`:
load_dotenv()

# --- Restante da aplicação ---
```

Outra maneira também seria pelo próprio script:
```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

Outra alternativa sáo as variáveis de ambiente:
```powershell
[System.Environment]::SetEnvironmentVariable("LANGSMITH_TRACING", "true", "User")
[System.Environment]::SetEnvironmentVariable("LANGSMITH_API_KEY", "your_api_key_here", "User")
```
É muito importante que a chave de API do modelo escolhido (ChatGpt, Gemini etc) estejam devidamente inseridas ou nas variáveis de ambiente ou mesmo configurada no script ou até mesmo através do documento .env como demonstrado nas alterantivas acima para que náo ocorram erros.

# Estrutura da Aplicação
A aplicação segue cinco passos:

1. O primeiro passo é uma função que define toda a lógica para o processamento dos arquivos que serão futuramente vetorizados e indexados em um diretório local. Primeiro, os arquivos são carregados do diretório onde se encontram, logo depois são contidos em um único objeto como uma unidade de texto. Ao final, essa unidade de texto é dividida em várias partes menores para que sejam futuramente transformados em embeddings.

![](/image/data_processing.png)

```python
# Definindo o diretório onde estão os arquivos pdf:
from langchain_community.document_loaders import PyPDFDirectoryLoader

directory = "../docs"  # Indicamos onde está o diretório com os arquivos desejamos processar.

# Instanciando a classe que usaremos para carregar nossos arquivos:
loaded = PyPDFDirectoryLoader(path=directory, glob="**/*.pdf")
docs = loaded.load() 
```
No exemplo acima, utilizamos o PyPDFDirectoryLoader, um dos componentes do módulo `document_loader` que é específico para carregar vários arquivos pdf de um diretório, este módulo possui várias opções para carregar arquivos contidos em diferentes fontes, `PyPDFLoader` por exemplo, é específico para carregar um arquivo pdf, `WebBaseLoader` é utilizado caso o objetivo é carregar arquivos de uma página da web etc. Existem vários outros componentes do `document_loader`, cada um específico para diversas fontes de arquivos, [aqui](https://python.langchain.com/docs/integrations/document_loaders/) é possível encontrar todas as opções.

Uma vez nossos arquivos carregados, é realizado o processo de split destes arquivos, ou seja, o nosso objeto contém uma lista de  objetos `Document`, onde cada um desses objetos contém os textos retirados de nossos arquivos pdf, agora, esses textos serão particionados em partes menores. Esse processo facilita na etapa de embedding e indexação.

```python
# Importando as dependências:
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Instanciando e estabelecendo o número de characteres que cada parte de texto terá:
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# Particionando:
all_splits = text_splitter.split_document(documents=docs)
```
Agora com os arquivos devidamente repartidos em partes menores, já estamos prontos para o próximo passo, embeddings e indexing.

2. O segundo passo define todo o processo de vetorização e indexação dos arquivos previamente carregados e divididos no primeiro passo. Nessa etapa é utilizado um modelo de llm para transformar cada pedaço de texto em embedding, um processo de atribuição de um valor numérico para cada pedaço de texto. Posteriormente, ocorre a indexação desses embeddings, ou seja, o armazenamento em banco de dados local (Nesse caso, foi utilizado o Chroma).
![](/image/indexing_embedding.png)

```python
# Dependências:
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Intanciando o modelo de embedding que vai gerar os vetores:
embeddings = GoogleGenerativeAIEmbeddings(model="Modelo do Seu LLM Aqui")

# Inicialiazando o banco de dados com o cliente Chroma:

# Diretório onde será inicializado o banco com os vetores:
diretorio_chromadb = "../vector_store"

vector_store = Chroma(collection_name="Nome da coleção", embedding_function=embeddings, persist_directory=diretorio_chromadb)

ids = vector_store.add_document(all_splits)  # Esta etapa indexa os vetores.
```

Na etapa exemplificada acima é descrevido como é feito o processo de instanciar o modelo de LLM escolhido, Google GenAI, ChatGpt, Claude etc, eles seráo os responsáveis por gerar os vetores para posterior indexação pelo Chroma. Em seguida, iniciamos o banco de dados com o Chroma no diretório indicado, pois nesse caso estamos armazenando os vetores/embeddings localmente, mas é perfeitamente possível utilizar outros clientes, como por exemplo, AstraDB, FAISS, MongoDB etc, a lista com todas as opções é encontrada [aqui.](https://python.langchain.com/docs/integrations/vectorstores/) Já sobre o modelos de embeddings disponíveis, podem ser econtrados neste [link.](https://python.langchain.com/docs/integrations/text_embedding/)

3. O terceiro passo define o processo de busca com base nas perguntas feitas pelo usuário. É realizada uma recuperação dos arquivos indexados no diretório escolhido no passo 2, e então esse resultado, chamado de contexto é passado para o quarto passo para gerar a resposta final ao usuário através de um modelo de linguagem.
![](/image/retieve_generation.png)

```python
# Exemplo input:
perunta = "Qual o objetivo do projeto?"
resposta = vector_store.similarity_search(pergunta)
print(f"Resposta: {resposta}")
```

```python
# Output:
Resposta: O projeto tem o intuito apenas de demonstrar o passo a passo de uma simples aplicação RAG (Retrieval-Augmented Generation). São utilizados os components dos frameworks LangChain, langGraph e alguns components integrados como Google GenAI para os passos de vetorização e generação do output, assim como também o Chroma client para indexição dos embeddings localmente (no caso deste projeto) e posteriormente o passo de recuperação.
```

Tanto a pergunta quanto a resposta são passadas para o LLM para gerar o output ao usuário, este passo é definido no passo seguinte. 

4. O quarto passo é onde definimos o prompt que será passada ao llm, onde estará tanto a pergunta do usuário quanto o contexto (o conteúdo recuperado dos nossos arquivos) para que seja gerado o output.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Definimos o template:
template = """
Your are an AI assistent for answering questions based on the provide context.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know, don't try to make up an  answer.
Keep the answer concise an to the point.

Context: {context}

Question: {question}

Answer:
"""

# Instanciamos o template e o LLM:
rag_template = ChatPromptTemplate.from_template(template)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
```

```python
# Definimos o prompt:
message = rag_template.invoke({"question": pergunta, "context": resposta})
response = llm.invoke(message)
print(f"Resposta: {response.content}")
```

5. O quinto e último passo é onde agrupamos todos os passos anteriores da nossa aplicação com o LangGraph. Nesta etapa, definimos a sequência de qual passo será realizado primero e onde finalizar.


# Strutura do projeto
```text
projeto-root/
├── /docs                      # Arquivos pdf ficam aqui.
├── /vector_store              # Chroma DB, onde armazenamos recuperamos os vetores.
├── /script/rag_app.py         # Script da aplicação 
```


# Demo
Neste caso demostrado abaixo, a aplicação foi inicializada no prórpio terminal com o comendo:
```
python rag_app.py
```
![](/image/rag_app_video.gif)
