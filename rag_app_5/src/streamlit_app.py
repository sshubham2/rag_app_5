import streamlit as st

home_page = st.Page("homepage.py", title="Home", icon="🏚️")
vec_db_mgnmnt = st.Page("vec_mng.py", title="Vector Database Management", icon="🛠️")
personality_mgnmnt = st.Page("personality_managemnt.py", title="Personality Management", icon="🛠️")
chatbot = st.Page("chatbot.py", title="Chatbot", icon="💬")
imgText = st.Page("imageChat.py", title="Image 2 Text", icon="📷")

pg = st.navigation([home_page, chatbot,imgText, vec_db_mgnmnt, personality_mgnmnt])
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout='wide')

pg.run()