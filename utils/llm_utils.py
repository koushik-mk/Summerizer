from langchain_community.chat_models import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

def load_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

def get_prompt_template():
    return ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
