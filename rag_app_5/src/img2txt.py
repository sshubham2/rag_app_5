# template.py

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from dotenv import load_dotenv
import os
from pathlib import Path

from config import ModelProvider, VisionModelProvider
from models import setup_anthropic_vision_model, setup_openai_vision_model, setup_ollama_vision_model
from vector_store import VectorStore
import logging
import base64
from io import BytesIO
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class RAGImageAppTemplate:
    def __init__(self, personality, prompt_name, prompt_name_rag):
        self.vStore = VectorStore()
        self.personality = personality
        self.prompt_name = prompt_name
        self.prompt_name_rag = prompt_name_rag
        self.initialize_session_state()

    def initialize_session_state(self):
        for key, default_value in [
            (f"selected_model_{self.personality}", None),
            (f"rag_{self.personality}_store", {}),
            (f"rag_{self.personality}_messages", []),
            (f"rag_{self.personality}_config", {"configurable": {"session_id": f"rag_{self.personality}_123"}}),
            (f"selected_vector_{self.personality}", None),
            (f"use_vector_db_{self.personality}", False),
            (f"image_description_{self.personality}", None),
            (f"image_processed_{self.personality}", False),
        ]:
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def encode_image(self, image):
        """Convert PIL Image to base64, handling different image formats"""
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def setup_vision_llm(self, provider):
        model_setup = {
            VisionModelProvider.OPENAI.value: setup_openai_vision_model,
            VisionModelProvider.ANTHROPIC.value: setup_anthropic_vision_model,
            VisionModelProvider.OLLAMA.value: setup_ollama_vision_model
        }
        return model_setup[provider]()

    def setup_sidebar(self):
        with st.sidebar:
            st.session_state[f"use_vector_db_{self.personality}"] = st.checkbox("Use Vector DB for Context", False)
            if st.session_state[f"use_vector_db_{self.personality}"]:
                available_vector_dbs = self.vStore.get_available_vector_dbs()
                if available_vector_dbs:
                    selected_vector = st.selectbox(
                        'Choose Vector DB:',
                        available_vector_dbs,
                        index=0,
                    )
                    # Update the selected vector store if it has changed
                    if selected_vector != st.session_state[f"selected_vector_{self.personality}"]:
                        st.session_state[f"selected_vector_{self.personality}"] = selected_vector
                        self.vStore.load_vector_store(selected_vector)
                        st.session_state.current_vector_db = selected_vector  # Track current vector DB
                        st.success(f"Vector DB updated to: {selected_vector}")  # Feedback to user
                        st.rerun()  # Rerun the script to reflect changes
                else:
                    st.warning('No Vector DB available. Please create a new one.')
            else:
                st.session_state[f"selected_vector_{self.personality}"] = None
                st.session_state.current_vector_db = None

        # Retrieve the current retriever based on user choice
        retriever = self.vStore.get_retriever() if st.session_state[f"use_vector_db_{self.personality}"] else None

        st.sidebar.header("Available LLM Model")
        selected_provider = st.sidebar.selectbox(
            "Choose Company:",
            [provider.value for provider in VisionModelProvider],
            index=0,
            key=f'llm_provider_{self.personality}'
        )

        return self.setup_vision_llm(selected_provider), retriever

    def get_image_description(self, image, llm):
        """Get image description using the selected model"""
        try:
            # Convert image to base64
            base64_image = self.encode_image(image)

            # Create a message with the image for analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please provide a detailed description of this image. Include all relevant details about what you see, including objects, people, text, colors, and any notable features."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]

            # Use the existing LLM setup to process the image
            response = llm.invoke(messages)
            return response.content

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
        
    def process_image_section(self, llm):
        """Handle image upload and processing in main window"""
        st.write("### 📷 Image Analysis")
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image to analyze",
                type=['png', 'jpg', 'jpeg'],
                key=f"image_uploader_{self.personality}"
            )
        
        # Reset image processed state when a new file is uploaded
        if uploaded_file is None:
            st.session_state[f"image_processed_{self.personality}"] = False
            return

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with col2:
            process_button = st.button("Analyze Image", key=f"process_image_{self.personality}")

            # Check if this is a new image by comparing with previous upload
            current_image_data = uploaded_file.getvalue()
            previous_image_data = st.session_state.get(f"previous_image_data_{self.personality}")

            is_new_image = (previous_image_data != current_image_data)

            if process_button and (is_new_image or not st.session_state[f"image_processed_{self.personality}"]):
                with st.spinner("Analyzing image..."):
                    # Store current image data for comparison
                    st.session_state[f"previous_image_data_{self.personality}"] = current_image_data

                    # Get image description using selected model
                    description = self.get_image_description(image, llm)

                    if description:
                        st.session_state[f"image_description_{self.personality}"] = description
                        st.session_state[f"rag_{self.personality}_messages"].append({
                            "role": "assistant",
                            "content": f"📸 Image Analysis:\n\n{description}\n\nYou can now ask questions about the image or discuss any insights from it."
                        })

                        st.session_state[f"image_processed_{self.personality}"] = True
                        st.success("Image analyzed successfully! You can now chat about it below.")

    def clear_chat(self):
        if st.sidebar.button('Clear Chat', key=f'clear_chat_{self.personality}'):
            st.session_state[f"rag_{self.personality}_messages"] = []
            st.session_state[f"rag_{self.personality}_store"] = {}
            st.rerun()

    def get_session_history(self,session_id: str):
        if session_id not in st.session_state[f"rag_{self.personality}_store"]:
            st.session_state[f"rag_{self.personality}_store"][session_id] = ChatMessageHistory()
        return st.session_state[f"rag_{self.personality}_store"][session_id]

    def setup_chain(self, llm, retriever):
        # Include image description in context if available
        image_context = f"\nRelevant Image Context: {st.session_state[f'image_description_{self.personality}']}" if st.session_state[f"image_description_{self.personality}"] else ""
        if st.session_state[f"use_vector_db_{self.personality}"] and retriever:
            # Use the existing RAG chain setup
            contextualize_q_system_prompt = """
                Given a chat history and the latest user question which might reference context in the chat history,
                formulate a standalone question which can be unserstood without the chat history.
                {image_context} 
                DO NOT answer the question, just reformulate it if needed and otherwise return its as is.
                """
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", self.prompt_name_rag),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            return RunnableWithMessageHistory(
                rag_chain,
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        else:
            # Use a simple chain without RAG
            qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.prompt_name}{image_context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ])

        chain = qa_prompt | llm
        return RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            )

    def display_chat_messages(self):
        for message in st.session_state[f"rag_{self.personality}_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self, conversational_chain):
        if prompt := st.chat_input("Hi! How can I help you?", key=f'chat_input_{self.personality}'):
            st.session_state[f"rag_{self.personality}_messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    if st.session_state[f"use_vector_db_{self.personality}"] : 
                        with st.spinner("Thinking🤔"):
                            response = conversational_chain.invoke({"input": prompt}, config=st.session_state[f"rag_{self.personality}_config"])
                            context_list = list(set(context.metadata['source'] for context in response['context']))
                            answer = f"{response['answer']}\n\n📌 Source: {', '.join(context_list)}"
                            response = st.write(answer)
                    else:
                        # When not using vector DB, response is an AIMessage
                        stream = conversational_chain.stream({"input": prompt}, config=st.session_state[f"rag_{self.personality}_config"])
                        response = st.write_stream(stream)
                except Exception as e:
                    st.info(f"An error occurred: {str(e)}. Please check your API key or try again.")
                    st.stop()
            st.session_state[f"rag_{self.personality}_messages"].append({"role": "assistant", "content": answer if st.session_state[f"use_vector_db_{self.personality}"] else response})


    def run(self, welcome_string):
        st.markdown(welcome_string)
        llm, retriever = self.setup_sidebar()
        self.process_image_section(llm)
        conversational_chain = self.setup_chain(llm, retriever)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_chain)