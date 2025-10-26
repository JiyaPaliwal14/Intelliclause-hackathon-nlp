import streamlit as st
import requests
import io

# --- Configuration ---
FASTAPI_BACKEND_URL = "http://localhost:8000/process-and-query/"
st.set_page_config(page_title="DocuBot", page_icon="📄", layout="centered")

# --- UI Elements ---
st.title("📄 IntelliClause: Intelligent Document Processing and Decision Explanation System")
st.markdown("Upload a legal document (PDF) and ask questions to get instant, context-aware answers.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and not st.session_state.document_processed:
        with st.spinner("Processing document... This may take a moment."):
            try:
                # Prepare file and data for the API request
                files = {'pdf_file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # In this simple example, we process the doc and ask a dummy question
                # to get it indexed. A better approach might be a dedicated /upload endpoint.
                initial_question_payload = {"questions": ["What is the main subject of this document?"]}
                
                # Send request to the backend
                response = requests.post(FASTAPI_BACKEND_URL, files=files, data=initial_question_payload)
                response.raise_for_status() # Raises an exception for 4XX/5XX errors
                
                result = response.json()
                
                # Store document info in session state
                st.session_state.document_id = result.get("document_id")
                st.session_state.uploaded_file_name = result.get("file_name")
                st.session_state.document_processed = True
                
                st.success(f"✅ Successfully processed '{st.session_state.uploaded_file_name}'!")
                st.info("You can now ask questions about the document in the chat window.")

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: Could not connect or process the document. Details: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

    if st.session_state.document_processed:
        st.success(f"'{st.session_state.uploaded_file_name}' is ready.")
        if st.button("Upload Another Document"):
            # Reset session state to allow for a new upload
            st.session_state.messages = []
            st.session_state.document_processed = False
            st.session_state.uploaded_file_name = None
            st.session_state.document_id = None
            st.experimental_rerun()


# --- Main Chat Interface ---

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.document_processed:
        st.warning("Please upload a document first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare the payload for the API
                    payload = {
                        "questions": [prompt],
                        "document_id": st.session_state.document_id
                    }
                    
                    # We don't need to send the file again, just the doc_id and question
                    # However, the current endpoint requires a file. Let's create a query-only endpoint.
                    # For now, let's adapt by resending a dummy file or modifying the backend logic.
                    # Let's assume we modify the backend to handle this case.
                    
                    # We need a dedicated query endpoint. For now, we will re-use the process endpoint
                    # This is inefficient but demonstrates the flow.
                    # A better design would be:
                    # 1. POST /upload -> processes PDF, returns doc_id
                    # 2. POST /query -> takes doc_id and question, returns answer
                    
                    # Since we only have one endpoint, we can't query without re-uploading.
                    # This is a limitation of the current backend design.
                    # Let's show the user an informative message.
                    
                    # A better way: Let's assume a dedicated /query endpoint.
                    # For this example, let's just show how it *would* work if the backend supported it.
                    # And then provide the code to make the backend support it.
                    
                    # Let's call a hypothetical /query endpoint
                    query_url = "http://localhost:8000/query/" # We will create this endpoint
                    query_payload = {"document_id": st.session_state.document_id, "question": prompt}
                    response = requests.post(query_url, json=query_payload)
                    response.raise_for_status()
                    
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except requests.exceptions.RequestException as e:
                    st.error(f"API Error during query: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during query: {e}")
