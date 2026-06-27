

class ScientistAgent():
    def __init__(self):
        self.prompt = ""
        self.memory = []
        self.plan = None

    def observe(self, input_data):
        """
        Observe the environment or user input.
        Returns a structured observation.
        """
        observation = f"Observation: {input_data}"
        self.memory.append(("Observe", observation))

    def think(self, observation: str):
        """
        Thinking about observation ;-)
        """
        thought = f"Thought: Based on '{observation}', I will generate  research plan."
        self.memory.append(("think", thought))
        return thought 
    
    def act(self, thought: str):
        """
        Acting :-)
        """
        self.plan = (
            "Research Plan:\n"
            "1. Identify key papers on the topic.\n"
            "2. Summarize core approaches.\n"
            "3. Highlight current limitations and open challenge.\n"
            "4. Propose a novel direction based on the findings."
        )

        action = f"Action: Generated a detailed plan.\n{self.plan}"
        self.memory.append(('act', action))
        return action
    