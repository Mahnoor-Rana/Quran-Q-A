
# import fitz  # PyMuPDF
# from transformers import AutoTokenizer, AutoModel
# import torch

# def load_pdf(file_path):
#     """Extracts text from a PDF file."""
#     doc = fitz.open(file_path)
#     text = ""
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         text += page.get_text()
#     return text

# def embed_document(text, tokenizer, model):
#     """Generates embeddings for the input text."""
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # Use the embeddings from the last hidden state
#     embeddings = outputs.last_hidden_state.mean(dim=1)
#     return embeddings

# # Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = AutoModel.from_pretrained("bert-base-multilingual-cased")
# # Example usage
# file_path = r"C:\Users\Mahnoor Rana\Desktop\AL-QURAN\Quran-Q-A\data\Quran-e-Pak.pdf"  # Replace with your PDF file path
# urdu_text = load_pdf(file_path)
# embedding = embed_document(urdu_text, tokenizer, model)

# print(len(embedding[0]))

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# import torch

# model_id = "large-traversaal/Alif-1.0-8B-Instruct"

# # Load tokenizer and model in CPU mode without quantization
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

# # Create text generation pipeline
# chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

# # Function to chat
# def chat(message):
#     response = chatbot(message, max_new_tokens=100, do_sample=True, temperature=0.3)
#     return response[0]["generated_text"]

# # Example chat
# user_input = "شہر کراچی کی کیا اہمیت ہے؟"
# bot_response = chat(user_input)

# print(bot_response)


# from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# # Load a fine-tuned XLM-RoBERTa model for QA
# model_name = "deepset/xlm-roberta-large-squad2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# # Create a question-answering pipeline
# qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# # Example usage
# context = "کراچی پاکستان کا سب سے بڑا شہر ہے۔"
# question = "پاکستان کا سب سے بڑا شہر کون سا ہے؟"
# result = qa_pipeline(question=question, context=context)

# print(f"Answer: {result['answer']}")

# import PyPDF2
# import torch
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# def extract_text_from_pdf(pdf_path):
#     """
#     Extract text from a PDF file
    
#     Args:
#         pdf_path (str): Path to the PDF file
    
#     Returns:
#         str: Extracted text from the PDF
#     """
#     try:
#         with open(pdf_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page in reader.pages:
#                 text += page.extract_text()
#         return text
#     except Exception as e:
#         print(f"Error extracting text from PDF: {e}")
#         return None

# def setup_qa_model(model_name="asadjaved/urdu-qa"):
#     """
#     Load Urdu Question Answering model and tokenizer
    
#     Args:
#         model_name (str): Hugging Face model identifier
    
#     Returns:
#         tuple: (tokenizer, model)
#     """
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
#         # Move to GPU if available
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         return tokenizer, model
#     except Exception as e:
#         print(f"Error loading QA model: {e}")
#         return None, None

# def answer_question(context, question, tokenizer, model):
#     """
#     Answer a question based on the given context
    
#     Args:
#         context (str): Text from which to extract answer
#         question (str): Question to be answered
#         tokenizer: Tokenizer for the model
#         model: Question Answering model
    
#     Returns:
#         str: Extracted answer
#     """
#     try:
#         # Prepare inputs
#         inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        
#         # Get model outputs
#         outputs = model(**inputs)
        
#         # Find the best answer span
#         start_scores = outputs.start_logits
#         end_scores = outputs.end_logits
        
#         start_index = torch.argmax(start_scores)
#         end_index = torch.argmax(end_scores)
        
#         # Decode the answer
#         answer_tokens = inputs['input_ids'][0][start_index:end_index+1]
#         answer = tokenizer.decode(answer_tokens)
        
#         return answer
#     except Exception as e:
#         print(f"Error answering question: {e}")
#         return None

# def main():
#     # Path to your Urdu PDF
#     pdf_path = 'your_urdu_document.pdf'
    
#     # Extract text from PDF
#     pdf_text = extract_text_from_pdf(pdf_path)
    
#     if pdf_text:
#         # Setup QA model
#         tokenizer, model = setup_qa_model()
        
#         if tokenizer and model:
#             # Example question
#             question = "آپ کی کتاب کس بارے میں ہے؟"
            
#             # Get answer
#             answer = answer_question(pdf_text, question, tokenizer, model)
            
#             print("Question:", question)
#             print("Answer:", answer)

# if __name__ == "__main__":
#     main()

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# def load_multilingual_model():
#     """
#     Load a multilingual model that can handle Urdu
#     """
#     try:
#         # Use multilingual models that support Urdu
#         model_name = "google/mt5-base"
        
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
#         # Set device
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
        
#         return tokenizer, model
    
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None, None

# def generate_text(tokenizer, model, prompt, max_length=100):
#     """
#     Generate text using the multilingual model
#     """
#     try:
#         # Ensure prompt is in Urdu
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         # Encode input
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
#         # Generate text
#         output = model.generate(
#             input_ids, 
#             max_length=max_length, 
#             num_return_sequences=1,
#             do_sample=True
#         )
        
#         # Decode and return generated text
#         return tokenizer.decode(output[0], skip_special_tokens=True)
    
#     except Exception as e:
#         print(f"Text generation error: {e}")
#         return None

# def main():
#     # Load multilingual model
#     tokenizer, model = load_multilingual_model()
    
#     if tokenizer and model:
#         # Example Urdu prompts
#         prompts = [
#             "آج کا دن بہت خوبصورت ہے",
#             "پاکستان کی معیشت",
#             "اسلام کے بنیادی اصول"
#         ]
        
#         # Generate text for each prompt
#         for prompt in prompts:
#             print("\nPrompt:", prompt)
#             generated_text = generate_text(tokenizer, model, prompt)
#             print("Generated Text:", generated_text)

# if __name__ == "__main__":
#     main()



# import torch
# from transformers import pipeline

# def generate_urdu_text():
#     """
#     Generate Urdu text using available models
#     """
#     try:
#         # Use a text generation pipeline
#         generator = pipeline('text-generation', model='gpt2')
        
#         # Example Urdu prompts
#         prompts = [
#             "آج کا دن بہت خوبصورت ہے",
           
#         ]
        
#         # Generate text for each prompt
#         for prompt in prompts:
#             print("\nPrompt:", prompt)
#             generated = generator(prompt, max_length=100, num_return_sequences=1)
#             print("Generated Text:", generated[0]['generated_text'])
    
#     except Exception as e:
#         print(f"Error generating text: {e}")

# # Run the generation
# generate_urdu_text()


# from transformers import AutoTokenizer, AutoModel
# import torch

# # Load model and tokenizer
# model_name = "intfloat/multilingual-e5-large-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Example Urdu document
# urdu_text = "یہ ایک مثال ہے کہ ہم اردو متن کو ایمبیڈنگ میں کیسے تبدیل کر سکتے ہیں۔"

# # Convert text to tokenized input
# inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)

# # Generate embeddings
# with torch.no_grad():
#     embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling

# print("Embedding shape:", embeddings.shape)  # Should be (1, 1024)

import os
import torch
import numpy as np
import faiss
import gc
import time
import re
from pathlib import Path
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ===== Configuration =====
PDF_FOLDER = "./data"  # Update this path to where your Quran PDFs are stored
FAISS_INDEX_PATH = "quran_faiss_index.bin"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GENERATION_MODEL = "bigscience/bloomz-1b1"  # Good balance of size and Urdu support
CHUNK_SIZE = 256
MAX_GENERATION_LENGTH = 150

# ===== Memory Management =====
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("Memory cleared")

# ===== Text Extraction Functions =====
def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF with proper RTL handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Use TEXT mode for better Unicode extraction
            page_text = page.get_text("text") 
            text += page_text + "\n\n"
        doc.close()
        return clean_arabic_text(text)
    except Exception as e:
        print(f"Error extracting {Path(pdf_path).name}: {e}")
        return ""

def clean_arabic_text(text):
    """Clean and normalize Arabic/Urdu text"""
    # Remove CID placeholders that indicate encoding issues
    text = re.sub(r'\(cid:[0-9]+\)', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    # Remove isolated diacritics
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670\u0674\u06D6-\u06ED]+')
    text = arabic_diacritics.sub('', text)
    
    return text

def extract_all_pdfs(folder_path):
    """Process all PDFs in a folder"""
    print(f"Processing PDFs from: {folder_path}")
    all_text = ""
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")
            pdf_text = extract_text_from_pdf(pdf_path)
            all_text += pdf_text + "\n\n"
    
    # Save the combined text for inspection
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(all_text)
    
    print(f"Extracted {len(all_text)} characters")
    return all_text

# ===== Text Processing Functions =====
def split_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks of specified size"""
    print("Splitting text into chunks...")
    # For Arabic/Urdu, split on periods or new lines for better context
    sentences = re.split(r'[۔،.؟!\n]', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence exceeds chunk size, start a new chunk
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# ===== Model Loading =====
def load_models():
    print("Loading embedding model...")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    embed_model = AutoModel.from_pretrained(EMBEDDING_MODEL).to("cpu")
    print("✅ Embedding model loaded")

    print("Loading generation model...")
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32
    ).to("cpu")
    print("✅ Generation model loaded")
    
    return embed_tokenizer, embed_model, gen_tokenizer, gen_model

# ===== Embedding Functions =====
def get_embedding(text, embed_tokenizer, embed_model):
    """Generate embeddings for a text chunk"""
    inputs = embed_tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=CHUNK_SIZE
    ).to("cpu")
    
    with torch.no_grad():
        model_output = embed_model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Handle single-dimension case
        if len(embeddings.shape) == 0:
            embeddings = np.array([embeddings])
        return embeddings.astype("float32")

# ===== FAISS Index Functions =====
def create_faiss_index(chunks, embed_tokenizer, embed_model):
    """Create a FAISS index from text chunks"""
    print("Generating embeddings...")
    embeddings = []
    
    # Process chunks with progress feedback
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"Processing chunk {i}/{len(chunks)}")
        embedding = get_embedding(chunk, embed_tokenizer, embed_model)
        embeddings.append(embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype("float32")
    
    # Create and save FAISS index
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index created and saved to {FAISS_INDEX_PATH}")
    
    return index

def load_or_create_index(text, embed_tokenizer, embed_model):
    """Load existing FAISS index or create new one"""
    chunks = split_text(text)
    
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing FAISS index...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print(f"Loaded index with {index.ntotal} vectors")
    else:
        print("Creating new FAISS index...")
        index = create_faiss_index(chunks, embed_tokenizer, embed_model)
    
    return index, chunks

# ===== Retrieval Functions =====
def retrieve_context(query, index, chunks, embed_tokenizer, embed_model, top_k=3):
    """Retrieve most relevant chunks for a query"""
    print(f"Retrieving context for: {query}")
    
    # Get query embedding
    query_embedding = get_embedding(query, embed_tokenizer, embed_model).reshape(1, -1)
    
    # Search index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get text chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]
    
    # Format as context
    context = "\n\n".join(retrieved_chunks)
    return context

# ===== Generation Functions =====
def generate_answer(query, context, gen_tokenizer, gen_model):
    """Generate an answer based on query and context"""
    print("Generating answer...")
    
    # Format prompt in Urdu
    prompt = f"""سوال: {query}

معلومات مددگار:
{context}

براہ کرم جواب اردو میں تفصیل سے دیں:"""


    # Tokenize
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cpu")
    
    # Generate
    print("Running generation...")
    gen_start = time.time()
    with torch.no_grad():
        output_ids = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_GENERATION_LENGTH,
            num_beams=2,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=gen_tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    
    # Decode
    output = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Generation completed in {gen_time:.2f} seconds")
    return output

# ===== Main RAG System =====
def setup_rag_system():
    """Set up the complete RAG system"""
    # Load models
    embed_tokenizer, embed_model, gen_tokenizer, gen_model = load_models()
    
    # Extract text from PDFs
    text = extract_all_pdfs(PDF_FOLDER)
    
    # Check if text was extracted successfully
    if not text or len(text) < 100:
        print("ERROR: Not enough text extracted from PDFs. Check your files.")
        print(f"Text preview: {text[:100]}")
        return None, None, None, None, None, None
    
    # Create chunks and index
    index, chunks = load_or_create_index(text, embed_tokenizer, embed_model)
    
    return text, chunks, index, embed_tokenizer, embed_model, gen_tokenizer, gen_model

def answer_question(query, rag_system):
    """Answer a question using the RAG system"""
    chunks, index, embed_tokenizer, embed_model, gen_tokenizer, gen_model = rag_system
    
    # Retrieve relevant context
    context = retrieve_context(query, index, chunks, embed_tokenizer, embed_model)
    
    # Generate answer
    answer = generate_answer(query, context, gen_tokenizer, gen_model)
    
    return answer, context

# ===== Main Execution =====
def main():
    print("Setting up Urdu Quran RAG system...")
    
    # Setup
    text, chunks, index, embed_tokenizer, embed_model, gen_tokenizer, gen_model = setup_rag_system()
    
    if not chunks or not index:
        print("Failed to set up RAG system")
        return
    
    rag_system = (chunks, index, embed_tokenizer, embed_model, gen_tokenizer, gen_model)
    
    # Interactive Q&A loop
    while True:
        user_query = input("\nEnter your question in Urdu (or 'exit' to quit): ")
        
        if user_query.lower() == 'exit':
            break
            
        start_time = time.time()
        answer, context = answer_question(user_query, rag_system)
        total_time = time.time() - start_time
        
        print("\n🤖 AI Answer:")
        print(answer)
        print(f"\n(Generated in {total_time:.2f} seconds)")
        
        # Optional: show the retrieved context
        show_context = input("\nShow retrieved context? (y/n): ")
        if show_context.lower() == 'y':
            print("\nContext used:")
            print(context)

if __name__ == "__main__":
    main()