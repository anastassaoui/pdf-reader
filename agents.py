from crewai import Agent
from langchain_groq import ChatGroq
import os

class CustomAgents:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="mixtral-8x7b-32768")

    def create_agent(self, role):
        descriptions = {
            "Market Analyst": "Specializes in market research for strategic planning.",
            "Marketing Strategist": "Focuses on developing actionable marketing strategies."
        }

        return Agent(
            role=role,
            backstory=descriptions[role],
            goal=f"Provide expert-level insights as a {role}.",
            verbose=True,
            llm=self.llm,
            max_rpm=30
        )
