
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

import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def setup_qa_model(model_name="asadjaved/urdu-qa"):
    """
    Load Urdu Question Answering model and tokenizer
    
    Args:
        model_name (str): Hugging Face model identifier
    
    Returns:
        tuple: (tokenizer, model)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading QA model: {e}")
        return None, None

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


from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
print(torch.cuda.is_available())

model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

urdu_docs = [
    "یہ ایک مثال ہے کہ ہم اردو متن کو ایمبیڈنگ میں کیسے تبدیل کر سکتے ہیں۔",
    "مشین لرننگ اور قدرتی زبان کی پروسیسنگ کا استعمال بہت عام ہو چکا ہے۔",
    "اردو میں ڈیٹا تلاش کرنے کے لئے ریکال ماڈل بہترین ثابت ہو سکتا ہے۔"
]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

embeddings = np.array([get_embedding(doc) for doc in urdu_docs])

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("✅ Urdu documents stored in FAISS!")


def retrieve_top_k(query, k=2):
    query_embedding = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return [urdu_docs[i] for i in indices[0]]

# Test retrieval
query = "مشین لرننگ کیا ہے؟"
retrieved_docs = retrieve_top_k(query)

print("🔍 Retrieved Urdu Documents:")
for doc in retrieved_docs:
    print("-", doc)

llm_name = "bigscience/bloom-1b7"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="cpu")

def generate_answer(query):
    context = "\n".join(retrieve_top_k(query, k=2))
    prompt = f"سوال: {query}\n\nیہ معلومات مددگار ہو سکتی ہیں:\n{context}\n\nجواب:"

    inputs = llm_tokenizer(prompt, return_tensors="pt").to("cpu")
    output = llm_model.generate(**inputs, max_new_tokens=200)
    return llm_tokenizer.decode(output[0], skip_special_tokens=True)

# Example
query = "اردو میں مشین لرننگ کے استعمالات کیا ہیں؟"
answer = generate_answer(query)
print("🤖 AI Answer:", answer)