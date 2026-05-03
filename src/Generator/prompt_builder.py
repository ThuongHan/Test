import yaml

# event here should be a structured output from the interpreter
# event =  { "What happened" :,
    #        "Why does it matter (globally and NL)" :,
    #        "Why does it matter for KickstartAI" :,
    #        "Key stance / opinion":,
    #        "Supporting arguments":
    #       }

class PromptBuilder:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
    
    def build(self, event: dict) -> str:
        voice = self.config["voice"]
        tone = self.config["tone"]
        style = self.config["writing_style"]
        output = self.config["output"]

        return f"""
VOICE:
{voice}

TONE: 
{tone}

WRITING RULES:
{style}

OUTPUT RULES:
{output}

EVENT:
{event}

TASK:
Generate LinkedIn posts according to all rules above.
"""



