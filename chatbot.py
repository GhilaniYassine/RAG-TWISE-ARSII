# chatbot.py

import os
from typing import Optional, List
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3.2:3b",  # Keep the model name that works via curl
        llm_temperature: float = 0.7,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "vector_db",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.

        Args:
            model_name (str): The HuggingFace model name for embeddings.
            device (str): The device to run the model on ('cpu' or 'cuda').
            encode_kwargs (dict): Additional keyword arguments for encoding.
            llm_model (str): The local LLM model name for ChatOllama.
            llm_temperature (float): Temperature setting for the LLM.
            qdrant_url (str): The URL for the Qdrant instance.
            collection_name (str): The name of the Qdrant collection.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Initialize Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Initialize Local LLM with enhanced configuration
        try:
            self.llm = ChatOllama(
                model=self.llm_model,
                temperature=self.llm_temperature,
                base_url="http://localhost:11434",
                verbose=True,
                format="json",  # Add format specification
                timeout=120,  # Increase timeout for longer responses
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
            )
            
            # Test model connection with simple prompt
            test_response = self.llm.invoke("Say 'test'")
            st.info(f"Model initialized successfully: {test_response}")
            
        except Exception as e:
            st.error(f"LLM Initialization Error: {str(e)}")
            st.info("Checking Ollama service...")
            # Add service check
            import subprocess
            try:
                subprocess.run(["curl", "http://localhost:11434/api/tags"], check=True)
            except subprocess.CalledProcessError:
                st.error("Ollama service not responding. Please check if it's running.")
            raise

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Initialize the Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,  # Set to False to return only 'result'
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response  # 'response' is now a string containing only the 'result'
        except Exception as e:
            st.error(f"⚠️ An error occurred while processing your request: {e}")
            return "⚠️ Sorry, I couldn't process your request at the moment."

    def chat(self, message: str) -> str:
        try:
            response = self.llm.predict(message)
            return response
        except Exception as e:
            return f"Error processing message: {str(e)}"

# Usage example
if __name__ == "__main__":
    try:
        print("🐺 Welcome to Yassine Ghilani's Timberwolves Chat System!")
        print("We're here to prove we deserve the TWICE challenge victory!")
        chatbot = ChatbotManager()
        
        # Check if LLM loaded correctly
        if not hasattr(chatbot, 'llm') or chatbot.llm is None:
            raise RuntimeError("LLM initialization failed")
            
        response = chatbot.chat("Hello!")
        st.success(f"🐺 Timberwolves Response: {response}")
    except RuntimeError as llm_error:
        st.error("🚫 LLM Loading Error: The language model failed to initialize")
        st.warning("Please check your API keys and model configurations")
        st.info("Contact Team Leader Yassine Ghilani for support at ghilaniyassine11@gmail.com")
    except Exception as e:
        st.error(f"🐺 Timberwolves Team Message: {str(e)}")
        st.info("Contact Team Leader Yassine Ghilani for support at ghilaniyassine11@gmail.com")
