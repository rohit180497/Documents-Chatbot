import streamlit as st
from streamlit_chat import message
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

class ChatbotApp:
    def __init__(self):
        self.history_key = "chat_history"
        self.generated_key = "generated_responses"
        self.history = []
        self.generated = ["👋 Hello! I'm your AI Assistant for Payments and Settlement Systems. Ask me anything."]
        self.past = ['Hey!']
        
        
        self.load_data()
        self.initialize_models()

    
    def load_data(self):
        # Load PDF files from the path
        loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        return text_chunks

    
    def initialize_models(self):
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                           model_kwargs={'device': "cpu"})

        # Create a vector store
        vector_store = FAISS.from_documents(self.load_data(), embeddings)

        # Create the Conversational Retrieval Chain
        llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                            model_type="llama",
                            config={'max_new_tokens': 158, 'temperature': 0.01})

        memory = ConversationBufferMemory(memory_key=self.history_key, return_messages=True)
        retriever = vector_store.as_retriever(search_kwargs={'k': 2})

        self.chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                                           retriever=retriever,
                                                           memory=memory)

    def conversation_chat(self, query):
        results = self.chain({"question": query, "chat_history": self.history})
        self.history.append((query, results['answer']))
        return results['answer']

    def initialize_session_state(self):
        if "history" not in st.session_state:
            st.session_state['history'] = self.history
        if "generated" not in st.session_state:
            st.session_state['generated'] = self.generated
        if "past" not in st.session_state:
            st.session_state['past'] = self.past

    def display_chat_history(self):
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask about payment related queries", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = self.conversation_chat(user_input)

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    def run(self):
        st.title("Enterprise Chatbot")
        self.initialize_session_state()
        self.display_chat_history()

# Instantiate and run the Streamlit app
if __name__ == "__main__":
    chatbot_app = ChatbotApp()
    chatbot_app.run()
