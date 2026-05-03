

from src.Generator.prompt_builder import PromptBuilder
from src.Generator.output_formatter import OutputFormatter
from src.utils.llm_clients import llm


class PostGenerator:
    def __init__(self):
        self.llm = llm()
        self.promt_builder = PromptBuilder()
        self.formatter = OutputFormatter()
    
    def generate(self, interpreted_event: dict) -> dict:
        prompt = self.promt_builder.build(interpreted_event)
        response = self.llm.invoke(prompt)

        return self.formatter.format(response)







