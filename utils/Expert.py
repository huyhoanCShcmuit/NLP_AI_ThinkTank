from typing import Dict, Any, Callable
from langchain.schema.messages import HumanMessage, AIMessage
from utils.Worker import Worker

class Expert(Worker):
    def __init__(self, model: str, expert_instruction: Dict[str, str]) -> None:
        super().__init__(model=model)
        self.expert_instruction = expert_instruction
        self.system_prompts = self.config["expert"]['system_prompts']
        self.examples = self.config["expert"]['examples']
    
    def format_debate_history(self, debate_history: list) -> str:
        """
        Formats the debate history into a coherent string.
        """
        history_text = ""
        for entry in debate_history:
            speaker = entry['role'].capitalize()
            history_text += f"{speaker}: {entry['content']}\n"

        print(history_text)
        return history_text

    def generate_argument(self, debate: Any, stream_handler: Callable) -> str:
        # Include the debate history in the human message
        debate_history_text = self.format_debate_history(debate.debate_history)

        system_prompt = self.system_prompts["system1"].replace("##debate_topic##", debate.topic)
        
        # RAG Pipeline: Load local FAISS index if it exists and perform vector retrieval
        import os
        context_text = ""
        if os.path.exists("faiss_index"):
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                
                # Load the FAISS vector database using the same Gemini embedding model
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=debate.api_key)
                vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                
                # Build query based on current topic and last message of debate history
                query = debate.topic
                if debate.debate_history:
                    query += " " + debate.debate_history[-1]["content"]
                
                # Retrieve top-k relevant chunks (grounded knowledge)
                docs = vector_store.similarity_search(query, k=3)
                if docs:
                    context_text = "\n\n[RETRIEVED KNOWLEDGE/CONTEXT FROM UPLOADED DOCUMENTS]:\n"
                    context_text += "\n---\n".join([doc.page_content for doc in docs])
                    context_text += "\n\nInstructions to Expert: Incorporate and cite facts from the RETRIEVED KNOWLEDGE above in your arguments when relevant to ground your answer and avoid hallucination."
            except Exception as e:
                # Fallback to normal execution if vector store loading fails
                print(f"Error loading or querying FAISS vector database: {e}")

        system_prompt += context_text
        system_prompt += "\n" + self.expert_instruction["instructions"] + "\n" + debate_history_text

        messages = [HumanMessage(content=system_prompt)]

        config = {
            "callbacks": [stream_handler]
        }

        for chunk in self.model.stream(messages, config=config):
            pass
