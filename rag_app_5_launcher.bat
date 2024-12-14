@echo off
REM location of the virtual environment
set VENV_PATH=.\.venv

REM Activate the virtual environment
call %VENV_PATH%\Scripts\activate.bat

REM Run the Streamlit app
streamlit run rag_app_5\src\streamlit_app.py