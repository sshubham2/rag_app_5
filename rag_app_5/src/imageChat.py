from img2txt import RAGImageAppTemplate
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import hashlib

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
Session = sessionmaker(bind=engine)

def load_personalities():
    session = Session()
    personalities = session.query(Personality).all()
    session.close()
    return personalities

with st.sidebar:
    personalities = load_personalities()
    personality_options = {p.personality_name: p for p in personalities}

    selected_personality_name = st.selectbox("Select Personality", options=list(personality_options.keys()))
    
    if selected_personality_name:
        selected_personality = personality_options[selected_personality_name]

        v_system_prompt = selected_personality.system_prompt
        v_system_prompt_rag = selected_personality.system_prompt_rag
        v_personality_title = selected_personality.personality_title
        hex_str = bytes(selected_personality_name, encoding='utf-8')
        session_hex = hashlib.sha256(hex_str).hexdigest()
        

app = RAGImageAppTemplate(session_hex,v_system_prompt, v_system_prompt_rag)
app.run(f" #### {v_personality_title} ")