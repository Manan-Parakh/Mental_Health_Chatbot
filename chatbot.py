import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever()

# Initialize chat template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a compassionate mental health chatbot. Your role is to provide support and understanding to users who may be experiencing emotional difficulties. Always maintain a warm, non-judgmental tone and encourage professional help when appropriate."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name="Llama3-8b-8192"
)

# Create RAG chain
rag_chain = create_history_aware_retriever(llm, retriever) | prompt | llm | StrOutputParser()

# Message store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Create conversational chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# Streamlit UI
st.set_page_config(page_title="Mental Health Support Chat", page_icon="🤗")
st.title("Mental Health Support Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Share what's on your mind..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Get bot response
    with st.spinner("Thinking..."):
        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": st.session_id}}
        )
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    with st.chat_message("assistant"):
        st.markdown(response["answer"])

# Sidebar with important information
with st.sidebar:
    st.title("Important Information")
    st.markdown("""
    ### About This Chat
    This is an AI-powered mental health support chatbot designed to provide a listening ear and emotional support. While it can offer helpful conversation, it is not a substitute for professional mental health care.
    
    ### Crisis Resources
    If you're experiencing a crisis or having thoughts of self-harm:
    
    🚨 **Emergency Services**: 911
    
    🆘 **National Crisis Hotline**: 988
    
    💭 **Crisis Text Line**: Text HOME to 741741
    
    ### Privacy Notice
    This chat is private and your conversations are not stored permanently. However, please avoid sharing personally identifiable information.
    
    ### Disclaimer
    This chatbot provides emotional support only. For professional mental health care, please consult with a licensed mental health professional.
    """)

