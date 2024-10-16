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

from config import ModelProvider
from models import setup_openai_model, setup_anthropic_model, setup_groq_model, setup_mistral_model, setup_embedding_model
from vector_store import VectorStore
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class RAGChatAppTemplate:
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
        ]:
            if key not in st.session_state:
                st.session_state[key] = default_value

    def setup_llm(self, provider):
        model_setup = {
            ModelProvider.OPENAI.value: setup_openai_model,
            ModelProvider.ANTHROPIC.value: setup_anthropic_model,
            ModelProvider.GROQ.value: setup_groq_model,
            ModelProvider.MISTRAL.value: setup_mistral_model
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
            [provider.value for provider in ModelProvider],
            index=0,
            key=f'llm_provider_{self.personality}'
        )

        return self.setup_llm(selected_provider), retriever

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
        if st.session_state[f"use_vector_db_{self.personality}"] and retriever:
            # Use the existing RAG chain setup
            contextualize_q_system_prompt = """
                Given a chat history and the latest user question which might reference context in the chat history,
                formulate a standalone question which can be unserstood without the chat history. DO NOT answer the question, just reformulate it if needed
                and otherwise return its as is.
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
            ("system", self.prompt_name),
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
        conversational_chain = self.setup_chain(llm, retriever)
        self.clear_chat()
        self.display_chat_messages()
        self.handle_user_input(conversational_chain)