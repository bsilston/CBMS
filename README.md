# rag-tutorial-v2
 
  RAG (Retrieval Augmented Generation) is applied to an LLM to enable the model to
  field queries about your own information / data. LLM's have built in natural language
  capabiliites but are trained on "general" information collected online. 

  RAG enables users to provide organization or lab specific data to the model, which is 
  then ingested by the model. The model is then able to answer queries specific to that
  information. 

#  RUNNING THE RAG APP

  We recommend created a new environment to execute the RAG program, as certain conflicts
  between libraries may emerge if using pytyon > 3.10

 To install dependencies: pip install -r requirements.txt (the app is updated from time to time and manual install of certain packages may be required)

 In addition to these dependencies, the mistral (or other) model is required. 
 To use the mistral model use: ollama pull mistral in a terminal / command prompt
 If you do not have Ollama go to their website and download the application
 
 To create a new environment (conda)
 conda create -n new_env_name python=3.10 -y
 where you replace new_env_name with any name of your choosing

 To activate the environment (make sure you use the name you chose above)
 conda activate new_env_name

 Once inside the environment, install dependencies (see above requirements)

  To run the app:
  streamlit run streamlit_app.py OR
  python -m streamlit run streamlit_app.py

  The command will open a browser window with the streamlit UI. From the UI you can drag
  a single PDF or zip file with [x] number of PDFs.

  Once the files are uploaded the UI will update. After it updates you may query in the 
  text box provided.
