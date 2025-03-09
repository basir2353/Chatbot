import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="LifeLine AI ", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
        .navbar {
            background-color: #2c3e50;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
        }
        .navbar h1 {
            color: white;
            font-size: 22px;
            margin: 0;
        }
        .footer {
            text-align: center;
            padding: 10px;
            margin-top: 30px;
        }
        .footer a {
            margin: 0 10px;
            text-decoration: none;
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="navbar"><h1> LifeLine AI – Chat for Health Advice</h1></div>', unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    """Loads FAISS vector store with embeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {str(e)}")
        return None


def set_custom_prompt():
    """Defines a stricter prompt to enhance answer relevance."""
    custom_prompt_template = """
    You are a health-focused AI assistant. Use only the provided medical documents as the main context.
    If the context does not contain enough information, provide general medical knowledge while clearly stating
    that this is not a medical diagnosis.

    Context: {context}
    Question: {question}

    Answer directly and clearly. If the question is about treatment, emphasize consulting a doctor.
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def load_llm(huggingface_repo_id, HF_TOKEN):
    """Loads the Hugging Face model endpoint with proper configuration."""
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            token=HF_TOKEN,
            temperature=0.3,
            model_kwargs={"max_length": 512}
        )
        return llm
    except Exception as e:
        st.error(f"❌ Error loading LLM: {str(e)}")
        return None


def format_sources(source_documents):
    """Formats sources to display titles, URLs, page numbers, and previews."""
    if not source_documents:
        return "**No sources found.**"

    formatted_sources = []
    for i, doc in enumerate(source_documents):
        title = doc.metadata.get("title", "Unknown Title")
        url = doc.metadata.get("url", "#")  # Fallback if URL is unavailable
        page_number = doc.metadata.get("page", "N/A")  # Extract page number
        snippet = (doc.page_content[:300] + "...") if doc.page_content else "No preview available"

        formatted_sources.append(
            f"**{i+1}. [{title}]({url}) (Page: {page_number})**\n> {snippet}\n"
        )

    return "\n".join(formatted_sources)


def main():
    """Main function to run the Streamlit chatbot."""
    st.title("Start Chat for Health Advice")

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # User input
    prompt = st.chat_input(" Ask a health-related question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', "content": prompt})

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        # Load FAISS and LLM
        vectorstore = get_vectorstore()
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

        if vectorstore is None or llm is None:
            st.error("⚠️ Unable to load required components.")
            return

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),  # Increase retrieval results
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response.get("result", "❌ No answer generated.")
            source_documents = response.get("source_documents", [])

            # Display response
            st.chat_message('assistant').markdown(f"** Answer:**\n{result}")

            if source_documents:
                with st.expander("📚 Sources Used"):
                    st.markdown(format_sources(source_documents))

            # Save message history
            st.session_state.messages.append({
                'role': 'assistant',
                "content": f"**🤖 Answer:**\n{result}\n\n📚 **Sources:**\n{format_sources(source_documents)}"
            })

        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")

    # ---- Footer with Social Media Links ----
    st.markdown("""
        <div class="footer">
            <p><strong>Developed by Abdul Basit</strong></p>
            <p>
                <a href="https://www.linkedin.com/in/abdul-basit-1a56b3275/" target="_blank">🔗 LinkedIn</a> |
                <a href="https://github.com/basir2353" target="_blank">🐙 GitHub</a> |
                <a href="https://www.instagram.com/dogar_basit08/" target="_blank">📷 Instagram</a> |
                <a href="https://www.facebook.com/mabdulbasit.dogar.1" target="_blank">📘 Facebook</a>
            </p>
            <p>📞 Contact: <a href="tel:+923469517653">+923469517653</a></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()