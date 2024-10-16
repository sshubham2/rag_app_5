import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy setup
Base = declarative_base()

class Personality(Base):
    __tablename__ = 'personalities'

    id = Column(Integer, primary_key=True)
    personality_name = Column(String(100), nullable=False)
    system_prompt = Column(Text, nullable=False)
    system_prompt_rag = Column(Text, nullable=False)
    personality_title = Column(String(100), nullable=False)

engine = create_engine('sqlite:///personalities.db', echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Database operations
def load_personalities():
    session = Session()
    personalities = session.query(Personality).all()
    session.close()
    return personalities

def add_personality(name, prompt, prompt_rag, title):
    session = Session()
    new_personality = Personality(
        personality_name=name,
        system_prompt=prompt,
        system_prompt_rag=prompt_rag,
        personality_title=title
    )
    session.add(new_personality)
    session.commit()
    session.close()

def update_personality(id, name, prompt, prompt_rag, title):
    session = Session()
    personality = session.query(Personality).get(id)
    if personality:
        personality.personality_name = name
        personality.system_prompt = prompt
        personality.system_prompt_rag = prompt_rag
        personality.personality_title = title
        session.commit()
    session.close()

def delete_personality(id):
    session = Session()
    personality = session.query(Personality).get(id)
    if personality:
        session.delete(personality)
        session.commit()
    session.close()

# Streamlit app
# st.set_page_config(page_title="Personality Management", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
nav = st.sidebar.radio("Go to", ["Add Personality", "Edit Personality"])

# Add Personality Page
if nav == "Add Personality":
    st.title('Add New Personality')

    new_name = st.text_input('Personality Name')
    new_title = st.text_input('Personality Title')
    st.write("System Prompt:")
    new_prompt = st.text_area(
        "Enter the system prompt here...",
        height=400,
        key="new_prompt"
    )
    st.write("System Prompt RAG:")
    new_prompt_rag = st.text_area(
        "Enter the system prompt RAG here...",
        height=400,
        key="new_prompt_rag"
    )

    if st.button('Add Personality'):
        add_personality(new_name, new_prompt, new_prompt_rag, new_title)
        st.success('Personality added successfully!')

# Edit Personality Page
elif nav == "Edit Personality":
    st.title('Edit Personality')

    personalities = load_personalities()
    personality_options = {p.personality_name: p for p in personalities}

    selected_personality_name = st.selectbox("Select a Personality to Edit", options=list(personality_options.keys()))

    if selected_personality_name:
        selected_personality = personality_options[selected_personality_name]

        edit_name = st.text_input('Personality Name', value=selected_personality.personality_name)
        edit_title = st.text_input('Personality Title', value=selected_personality.personality_title)

        # st.write("System Prompt:")
        edit_prompt = st.text_area(
            "System Prompt:",
            value=selected_personality.system_prompt,
            height=400,
            key=f"edit_prompt_{selected_personality.id}"
        )

        # st.write("System Prompt RAG:")
        edit_prompt_rag = st.text_area(
            "System Prompt RAG:",
            value=selected_personality.system_prompt_rag,
            height=400,
            key=f"edit_prompt_rag_{selected_personality.id}"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button('Save Changes'):
                update_personality(selected_personality.id, edit_name, edit_prompt, edit_prompt_rag, edit_title)
                st.success('Personality updated successfully!')
        with col2:
            if st.button('Delete Personality'):
                delete_personality(selected_personality.id)
                st.success('Personality deleted successfully!')
                st.rerun()