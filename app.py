import time
import re
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from langchain_community.llms import Ollama
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from typing import List

start_time = time.time()

def parse_quran_verses(documents: List[Document]):
    all_nodes = []
    
    for doc in documents:
        content = doc.text
        verses = re.split(r'(\d+:\d+\.|\d+\.)', content)
        
        for i, part in enumerate(verses):
            if re.match(r'^\d+:\d+\.$|\d+\.$', part):
                verse_num = part.strip()
                if i+1 < len(verses):
                    verse_text = verses[i+1].strip()
                    
                    if verse_text:
                        node = TextNode(
                            text=f"{verse_num} {verse_text}",
                            metadata={
                                "verse_num": verse_num,
                                "source": doc.metadata.get("file_path", "")
                            }
                        )
                        all_nodes.append(node)
    
    return all_nodes

try:
    print("Starting document loading...")
    input_dir = r'C:\Users\Mahnoor Rana\Desktop\AL-QURAN\data'
    loader = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".pdf"],
        recursive=True
    )
    raw_docs = loader.load_data()
    print(f'Loaded {len(raw_docs)} documents in {time.time() - start_time:.2f} seconds')
    
    print("Parsing documents into verse-based chunks...")
    verse_nodes = parse_quran_verses(raw_docs)
    print(f"Created {len(verse_nodes)} verse-based chunks")
    
    print("Initializing embedding model...")
    embed_time = time.time()
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        trust_remote_code=True
    )
    Settings.embed_model = embed_model
    print(f"Embedding model initialized in {time.time() - embed_time:.2f} seconds")
    
    print("Creating vector index from verse nodes...")
    index_time = time.time()
    index = VectorStoreIndex(verse_nodes)
    print(f"Index created in {time.time() - index_time:.2f} seconds")
    
    print("Initializing LLM...")
    llm_time = time.time()
    llm = Ollama(
        model="deepseek-r1:1.5b",
        temperature=0.1,
        timeout=240
    )
    Settings.llm = llm
    print(f"LLM initialized in {time.time() - llm_time:.2f} seconds")
    
    retriever = index.as_retriever(streaming=False, similarity_top_k=4)
    
    print("Setting up query engine...")
    qa_prompt_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    
    response_synthesizer = CompactAndRefine(
        text_qa_template=qa_prompt_tmpl,
        streaming=False
    )
    
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    
    print("Running query...")
    query_time = time.time()
    query_str = 'command of NIkkah For women?'
    
    retrieved_nodes = retriever.retrieve(query_str)
    
    print("\n=== RETRIEVED CHUNKS ===")
    for i, node in enumerate(retrieved_nodes):
        print(f"\nCHUNK {i+1} (Score: {node.score:.4f}):")
        print("-" * 40)
        print(f"Verse: {node.metadata.get('verse_num', 'Unknown')}")
        print(node.text[:500] + "..." if len(node.text) > 500 else node.text)
        print("-" * 40)
    print("=== END OF CHUNKS ===\n")
    
    response = query_engine.query(query_str)
    print(f"Query completed in {time.time() - query_time:.2f} seconds")
    print("\n=== FINAL RESPONSE ===")
    print(response)

except Exception as e:
    print(f"An error occurred: {e}")
