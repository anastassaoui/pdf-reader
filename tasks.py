from crewai import Task

class CustomTasks:
    def create_task(self, agent, business_type, task_type):
        task_descriptions = {
            "market_analysis": (
                f"Conduct a market analysis for {business_type}, focusing on competitors, market size, and trends."
            ),
            "marketing_strategy": (
                f"Develop a marketing strategy for {business_type} targeting key demographics."
            ),
        }

        expected_outputs = {
            "market_analysis": "Detailed market report for {business_type}.",
            "marketing_strategy": "Actionable marketing strategy document."
        }

        return Task(
            description=task_descriptions[task_type],
            agent=agent,
            expected_output=expected_outputs[task_type]
        )
