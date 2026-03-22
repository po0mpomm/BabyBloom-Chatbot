import os
from dotenv import load_dotenv

load_dotenv()

class BabyBloomEngine:
    def __init__(self, index_path="faiss_direct_index"):
        # Move heavy imports inside to speed up initial server boot
        from langchain_huggingface import HuggingFaceEmbeddings
        
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_path = index_path
        self.db = None
        self.llm = None
        self.rag_chain = None
        self.intent_classifier = None
        self._initialize()

    def _initialize(self):
        from langchain_community.vectorstores import FAISS
        from langchain_groq import ChatGroq
        from langchain.chains import create_history_aware_retriever, create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_cohere import CohereRerank
        from langchain.retrievers import ContextualCompressionRetriever

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Vector database not found at {self.index_path}")
        
        self.db = FAISS.load_local(self.index_path, self.embedding_function, allow_dangerous_deserialization=True)
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant")
        
        base_retriever = self.db.as_retriever(search_kwargs={"k": 15})
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        
        # 1. Recontextualization Chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, formulate a standalone question "
            "which can be understood without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, compression_retriever, contextualize_q_prompt
        )
        
        # 2. QA Chain
        qa_system_prompt = """
        **You are Baby Blooms, an AI Neonatal Information Assistant.** Your persona is empathetic, knowledgeable, and extremely focused on safety. Your expertise comes *exclusively* from the medical textbook excerpts provided in the **CONTEXT**.

        **Your primary goal is to structure your answer in a specific, multi-part format to ensure clarity and safety.**

        **CRITICAL INSTRUCTIONS:**
        1.  **Persona and Tone:** Maintain a calm, supportive, and professional tone.
        2.  **Grounding:** Your entire answer MUST be based *only* on the information within the **CONTEXT**.
        3.  **No Mention of "Context":** NEVER mention the words 'context', 'provided text', or 'snippets'.
        4.  **Handling "I Don't Know":** If the **CONTEXT** does not contain the answer, you MUST respond *only* with: "I've carefully checked the medical textbooks, but I couldn't find specific information on this topic. For your child's safety and well-being, it's very important to consult with a pediatrician for guidance."
        5.  **MANDATORY ANSWER STRUCTURE (Parts 1-3):** Empathetic acknowledgement + Potential causes, then Potential solutions, then Recommendation to consult a professional.

        ---
        CONTEXT:
        {context}
        ---
        """
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        self.intent_classifier = self._create_intent_classifier_chain(self.llm)

    def _create_intent_classifier_chain(self, llm):
        from langchain_core.prompts import PromptTemplate
        from langchain.chains import LLMChain
        intent_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are an intent classifier. Respond with only one of: 'conversational_greeting', 'meta_query', or 'informational_query'.
            User message: {question}
            Intent:"""
        )
        return LLMChain(llm=llm, prompt=intent_prompt)

    def ask(self, question, chat_history=[]):
        intent_response = self.intent_classifier.invoke({"question": question})
        intent = intent_response.get('text', 'informational_query').strip().lower()

        if "conversational_greeting" in intent:
            result = "Hello! I'm Baby Blooms, an AI assistant ready to help with your questions about newborn health. How can I assist you today?"
        elif "meta_query" in intent:
            result = "I'm Baby Blooms, an AI assistant designed to answer questions based on information from medical textbooks. How can I help with a newborn health question?"
        else: 
            from langchain_core.messages import HumanMessage, AIMessage
            # Format chat history for the modern chain
            formatted_history = []
            for msg in chat_history:
                if isinstance(msg, dict):
                    if msg.get('role') == 'user':
                        formatted_history.append(HumanMessage(content=msg.get('content')))
                    else:
                        formatted_history.append(AIMessage(content=msg.get('content')))
                # Handle cases where msg might already be a message object if called from other places
            
            response = self.rag_chain.invoke({
                "input": question, 
                "chat_history": formatted_history
            })
            result = response.get('answer', "I apologize, an error occurred.")
        
        disclaimer = "\n\n---\n**Disclaimer:** I am an AI assistant. This information is for educational purposes only and is not a substitute for professional medical advice."
        if "informational_query" in intent and "I couldn't find specific information" not in result:
            result += disclaimer
            
        return result, intent
