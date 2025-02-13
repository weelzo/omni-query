import streamlit as st
from modules.pdf_processor import extract_text_and_images
from modules.ocr import extract_text_from_image
from modules.embeddings import get_text_embeddings, get_image_embeddings, clip_model, clip_processor
from modules.vector_db import VectorDB
from modules.llm import generate_response
from modules.image_processor import process_images
import os
import gc
import tempfile

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for white theme and dark text
st.markdown('''
<style>
    /* Set the entire app background to white */
    .stApp {
        background-color: white !important;
    }

    /* Set all text to dark */
    body, .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown ol, .stMarkdown ul {
        color: #333333 !important;
    }

    /* Chat message text */
    .stChatMessage p {
        color: #333333 !important;
    }

    /* Expander text */
    .stExpander p {
        color: #333333 !important;
    }

    /* Alert text */
    .stAlert p {
        color: #333333 !important;
    }

    /* Input text */
    .stTextInput input {
        color: #333333 !important;
    }

    /* Custom background for chat messages */
    .stChatMessage {
        background-color: white !important;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* Custom background for user messages */
    .stChatMessage.user {
        background-color: #f0f7ff !important;
    }

    /* Custom background for assistant messages */
    .stChatMessage.assistant {
        background-color: white !important;
    }

    /* Custom styling for the header */
    .header {
        text-align: center;
        padding: 2rem 0;
    }

    .header h1 {
        color: #1E3D59;
        margin-bottom: 1rem;
    }

    .header p {
        color: #333333;
        font-size: 1.2em;
    }

    /* Custom styling for the upload section */
    .upload-section {
        background-color: white !important;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .upload-section h3 {
        color: #1E3D59;
        margin-bottom: 1rem;
    }

    /* Custom styling for the chat interface */
    .chat-section {
        background-color: white !important;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }

    .chat-section h3 {
        color: #1E3D59;
        margin-bottom: 1rem;
    }

    /* Custom styling for source text expander */
    .source-text {
        margin-bottom: 1rem;
    }

    .source-text .page {
        color: #1E3D59;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .source-text .content {
        background-color: white !important;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1E3D59;
        color: #333333 !important;
    }
</style>
''', unsafe_allow_html=True)

# Header with custom styling
st.markdown('''
<div class="header">
    <h1>📚 OmniQuery</h1>
    <p>Your AI-Powered Document Analysis Assistant</p>
</div>
''', unsafe_allow_html=True)

# File upload section with custom styling
st.markdown('''
<div class="upload-section">
    <h3>📄 Upload Your Document</h3>
</div>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type="pdf")
if uploaded_file is not None:
    # Create a temporary file that will be automatically cleaned up
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
        temp_pdf.write(uploaded_file.getbuffer())
        temp_path = temp_pdf.name

    try:
        # Extract text and images
        text_data, image_data = extract_text_and_images(temp_path)
    finally:
        # Clean up the temporary PDF file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Show processing success message
    st.success("✅ Document processed successfully! You can now ask questions about it.")

    # Generate embeddings
    text_embeddings = get_text_embeddings([chunk["text"] for chunk in text_data])
    image_embeddings = get_image_embeddings([img["path"] for img in image_data])

    # Initialize vector DBs with their respective dimensions
    text_dim = text_embeddings.shape[1]
    image_dim = image_embeddings.shape[1]
    
    # Store data in session state to prevent recreation on each rerun
    if 'text_db' not in st.session_state:
        st.session_state.text_db = VectorDB(text_dim)
    if 'image_db' not in st.session_state:
        st.session_state.image_db = VectorDB(image_dim)
    if 'image_data' not in st.session_state:
        st.session_state.image_data = []
    
    # Clear existing data and add new embeddings
    st.session_state.text_db.reset()
    st.session_state.text_db.add(text_embeddings)
    st.session_state.image_db.reset()
    st.session_state.image_db.add(image_embeddings)
    
    # Store image data in session state
    st.session_state.image_data = image_data
    
    # Clean up large numpy arrays we don't need anymore
    del text_embeddings
    del image_embeddings
    gc.collect()

    # Chat interface section
    st.markdown('''
    <div class="chat-section">
        <h3>💬 Chat with Your Document</h3>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display chat history with custom styling
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f'''<div style="background-color: {'#f0f7ff' if message['role'] == 'user' else 'white'}; 
                padding: 1rem; border-radius: 5px; margin-bottom: 0.5rem; color: #333;">{message['content']}</div>''', 
                unsafe_allow_html=True)

    # Query input with custom placeholder
    query = st.chat_input("💭 Ask any question about your document...")
    if query:
        # Get text embeddings for the query
        query_text_emb = get_text_embeddings([query])[0]
        
        # Search text content
        text_indices = st.session_state.text_db.search(query_text_emb)
        
        # Get CLIP text embeddings for image search
        clip_text_emb = clip_model.get_text_features(**clip_processor(text=[query], return_tensors="pt", padding=True)).detach().numpy()
        
        # Search images using CLIP embeddings
        image_indices = st.session_state.image_db.search(clip_text_emb[0])
        
        # Clean up query embedding after search
        del query_text_emb
        gc.collect()
        
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Get relevant text chunks and sort by page number
        relevant_chunks = [text_data[idx] for idx in text_indices]
        relevant_chunks.sort(key=lambda x: (x['page'], x['bbox'][1]))  # Sort by page and vertical position
        
        # Combine nearby chunks from the same page
        combined_chunks = []
        current_page = -1
        current_text = []
        
        for chunk in relevant_chunks:
            if chunk['page'] != current_page:
                if current_text:  # Save accumulated text from previous page
                    combined_chunks.append({
                        'page': current_page,
                        'text': ' '.join(current_text)
                    })
                current_page = chunk['page']
                current_text = [chunk['text']]
            else:
                current_text.append(chunk['text'])
        
        # Add the last chunk
        if current_text:
            combined_chunks.append({
                'page': current_page,
                'text': ' '.join(current_text)
            })
        
        # Format text for display and context
        relevant_text = []
        for chunk in combined_chunks:
            formatted_text = f"Page {chunk['page']}: {chunk['text']}"
            relevant_text.append(formatted_text)
        
        # Display source text in a styled expander
        with st.expander("📍 View Source Text"):
            if relevant_text:
                for chunk in relevant_text:
                    page, content = chunk.split(':', 1)
                    st.markdown(f'''
                    <div class="source-text">
                        <div class="page">{page}</div>
                        <div class="content">{content}</div>
                    </div>''', unsafe_allow_html=True)
            else:
                st.warning("No relevant text found in the document.")
                
        # Get relevant images
        relevant_images = [st.session_state.image_data[idx] for idx in image_indices]
        processed_images = process_images([img["path"] for img in relevant_images])
        
        # Generate context for LLM
        context = "\n".join(relevant_text)
        
        # Generate and display response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(query, context, processed_images)
                
                # Display response text
                st.markdown(response)
                
                # Display relevant images
                if relevant_images:
                    st.markdown("### Related Images")
                    cols = st.columns(min(3, len(relevant_images)))
                    for idx, (img, col) in enumerate(zip(relevant_images, cols)):
                        with col:
                            st.image(img["path"], caption=f"Image {idx + 1}", use_container_width=True)
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})