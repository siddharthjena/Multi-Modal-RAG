import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import re
import os
import numpy as np
import google.generativeai as genai
import tempfile
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams

# Ensure necessary NLP libraries are installed and stopwords downloaded
nltk.download("stopwords")

# Configure Google Generative AI
GOOGLE_API_KEY = "AIzaSyDOKm5KYY6LjLa20IbZg027fQauwyMOKWQ"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Create directories for storing data
INDEX_DIR = "faiss_indexes"
METADATA_DIR = "metadata_store"
MARKDOWN_DIR = "markdown_files"
for dir_path in [INDEX_DIR, METADATA_DIR, MARKDOWN_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def initialize_models():
    """Initialize all required models"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_index():
    """Create FAISS index"""
    text_embedding_dim = 384  # Dimension for MiniLM model
    return faiss.IndexFlatL2(text_embedding_dim), []

def save_markdown(md_text, pdf_name, key_terms):
    """Save markdown content to file with key terms and frequencies"""
    base_name = os.path.splitext(pdf_name)[0]
    safe_name = re.sub(r'[^\w\-_.]', '_', base_name)
    md_path = os.path.join(MARKDOWN_DIR, f"{safe_name}.md")
    
    with open(md_path, "w", encoding='utf-8') as f:
        f.write(md_text)
        # Append key terms and their frequencies as metadata
        f.write("\n\n# Key Terms and Frequencies\n")
        for term, freq in key_terms.items():
            f.write(f"- {term}: {freq}\n")
    
    return md_path

def save_index_and_metadata(index, metadata, pdf_name):
    """Save FAISS index and metadata to disk"""
    base_name = os.path.splitext(pdf_name)[0]
    safe_name = re.sub(r'[^\w\-_.]', '_', base_name)
    
    index_path = os.path.join(INDEX_DIR, f"{safe_name}_index.faiss")
    faiss.write_index(index, index_path)
    
    metadata_path = os.path.join(METADATA_DIR, f"{safe_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return index_path, metadata_path

def load_index_and_metadata(pdf_name):
    """Load FAISS index and metadata from disk"""
    base_name = os.path.splitext(pdf_name)[0]
    safe_name = re.sub(r'[^\w\-_.]', '_', base_name)
    
    index_path = os.path.join(INDEX_DIR, f"{safe_name}_index.faiss")
    metadata_path = os.path.join(METADATA_DIR, f"{safe_name}_metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    return None, None

def embed_text(text, text_embedder):
    """Embed text using SentenceTransformer"""
    return text_embedder.encode(text)

def parse_md_content(md_text, chunk_size=5):
    """Parse markdown content into larger blocks."""
    content_blocks = []
    current_block = []
    lines = md_text.splitlines()
    
    for line in lines:
        stripped_line = line.strip()
        
        if not stripped_line:
            continue
        
        if not (stripped_line.startswith("![") and "](" in stripped_line) and not stripped_line.startswith("["):
            current_block.append(stripped_line)
        
        if len(current_block) >= chunk_size:
            content_blocks.append({
                "type": "text",
                "content": " ".join(current_block)
            })
            current_block = []
    
    if current_block:
        content_blocks.append({
            "type": "text",
            "content": " ".join(current_block)
        })
    
    return content_blocks

from llama_parse import LlamaParse

def extract_key_terms(content, num_topics=5):
    """Identify key terms using N-grams, TF-IDF, and topic modeling."""
    # Tokenize and clean the content
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b\w+\b', content.lower())
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Generate N-grams
    n_grams = list(ngrams(words, 2)) + list(ngrams(words, 3))
    n_gram_counts = Counter([" ".join(gram) for gram in n_grams]).most_common(10)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_matrix = vectorizer.fit_transform([content])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray().flatten()))
    
    # Topic Modeling
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(tfidf_matrix)
    topic_terms = {}
    for idx, topic in enumerate(lda.components_):
        terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        topic_terms[f"Topic {idx+1}"] = terms
    
    # Aggregate key terms from all sources
    key_terms = dict(n_gram_counts)
    key_terms.update(tfidf_scores)
    for topic, terms in topic_terms.items():
        for term in terms:
            key_terms[term] = key_terms.get(term, 0) + 1  # Increment term frequency if already in key terms
    
    return key_terms

def process_pdf_with_llama(pdf_file):
    """Process PDF using LlamaParse and extract key terms"""
    try:
        existing_index, existing_metadata = load_index_and_metadata(pdf_file.name)
        if existing_index is not None:
            st.info("Loading existing index and metadata for this Document...")
            return existing_index, existing_metadata

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            pdf_path = tmp_file.name

        parser = LlamaParse(
            api_key="llx-rprJeq8dEhN3EJke5UDGSSWP1za8GV7540ci1yX8arJmRyWr",
            result_type="markdown",
            verbose=True,
            images=True,
            premium_mode=True
        )

        parsed_documents = parser.load_data(pdf_path)
        full_text = '\n'.join([doc.text for doc in parsed_documents])

        # Extract key terms
        key_terms = extract_key_terms(full_text)
        st.info(f"Key terms extracted: {key_terms}")

        # Save markdown with key terms
        md_path = save_markdown(full_text, pdf_file.name, key_terms)
        st.info(f"Saved markdown content to {md_path}")

        faiss_index, metadata_store = create_index()

        texts_to_embed = []
        for doc in parsed_documents:
            texts_to_embed.append(doc.text)
            metadata_store.append({"content": doc.text})

        if texts_to_embed:
            text_embedder = initialize_models()
            embeddings = text_embedder.encode(texts_to_embed)
            faiss_index.add(np.array(embeddings).astype('float32'))

        index_path, metadata_path = save_index_and_metadata(faiss_index, metadata_store, pdf_file.name)
        st.info(f"Saved index to {index_path} and metadata to {metadata_path}")

        os.unlink(pdf_path)

        return faiss_index, metadata_store

    except Exception as e:
        st.error(f"Error processing Document: {str(e)}")
        return None, None

def search_similar_content(query, text_embedder, faiss_index, metadata_store, k=10):
    query_embedding = embed_text(query, text_embedder)
    distances, indices = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
    
    results = []
    scores = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata_store):
            results.append(metadata_store[idx]["content"])
            scores.append(1 / (1 + dist))
    
   
    
    filtered_results = [
        content for content, score in zip(results, scores)
        if score > 0.2
    ]
    
    return filtered_results if filtered_results else []

# Function to generate a response
# Function to generate a response
# Function to generate a response
def generate_response(query, context, genai_model):
    if not context:
        return "I cannot answer this question based on the provided document."
    
    # Format history (with last 3 Q&A)
    history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.qa_history[-3:]])
    full_context = f"{history_text}\n\n{context}"
    
    # Construct prompt for generating response
    prompt = f"""
    Based ONLY on the following context, answer the question. If relevant details are not fully available, provide the information that is present and kindly note any specific information that is missing. Be helpful by mentioning related information that may assist in answering the question, and offer to expand on available details if useful. Additionally, provide quantitative details where needed, such as:

    The frequency of key terms (N-grams) in the content.
    The importance score of terms as determined by TF-IDF.
    Topic distribution, including which topics are most associated with particular terms or concepts.
    Quantified occurrences of specific term pairs, trigrams, or topics, wherever relevant.
    
    Provide 3 follow up questions from the context after every response.

    Context: {full_context}
    History of previous questions and answers:
    {history_text}
    Question: {query}

    Answer:"""

   
    
    # Get the response from the model
    response = genai_model.generate_content(prompt)
    answer = response.text

    # Update question-answer history
    st.session_state.qa_history.append((query, answer))

    return answer

# Streamlit UI
st.title("Multi-Modal RAG Chatbot")

# Initialize session state if not already initialized
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

if 'question_data' not in st.session_state:
    st.session_state.question_data = {}  # To store each question's additional info separately

# Upload file input
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "csv","pptx"])

if uploaded_file is not None and not st.session_state.get('processed', False):
    with st.spinner("Processing Document... This may take a few minutes."):
        try:
            text_embedder = initialize_models()
            st.session_state.text_index, st.session_state.text_metadata = process_pdf_with_llama(uploaded_file)

            if st.session_state.text_index is not None:
                st.session_state.text_embedder = text_embedder
                st.session_state.processed = True
                st.success("PDF processed successfully!")
                st.info(f"Created FAISS index with {st.session_state.text_index.ntotal} embeddings")
            else:
                st.error("Failed to process the Document")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Question asking section
if st.session_state.get('processed', False):
    st.subheader("Ask questions about your document")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Searching for relevant content..."):
            try:
                # Search for similar content based on the question
                similar_contents = search_similar_content(
                    query,
                    st.session_state.text_embedder,
                    st.session_state.text_index,
                    st.session_state.text_metadata
                )

                # Generate a response for the question
                response = generate_response(query, "\n".join(similar_contents), model)

                # Display the generated response
                st.write("Response:", response)

                # Display relevant content
                with st.expander("Relevant content"):
                    for content in similar_contents:
                        st.write(content)

                # **Logic to handle additional info separately per question**
                if query not in st.session_state.question_data:
                    st.session_state.question_data[query] = ""  # Initialize if not already set
                
                # Display text area only for current question
                additional_info = st.text_area(f"Provide additional details for your question: '{query}'", value=st.session_state.question_data[query])

                if additional_info:
                    # Update the additional information for the current question only
                    st.session_state.question_data[query] = additional_info
                    updated_query = f"{query} {additional_info}"

                    # Generate updated response based on additional info
                    with st.spinner("Generating updated response..."):
                        updated_response = generate_response(updated_query, "\n".join(similar_contents), model)

                    # Display the updated response
                    st.write("Updated Response:", updated_response)

            except Exception as e:
                st.error(f"An error occurred while searching: {str(e)}")
