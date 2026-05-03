import yaml
from src.utils.few_shot import FewShotPost
from src.utils.embedder import Embedder
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
    
    def build(self, event: dict, few_shot_posts: list[dict]) -> str:
        voice = self.config["voice"]
        tone = self.config["tone"]
        style = self.config["writing_style"]
        output = self.config["output"]

        examples = "\n\n".join(
            f"EXAMPLE {i+1}: \n{post["text"]}"
            for i, post in enumerate(few_shot_posts)
        )
        return f"""
VOICE:
{voice}

TONE: 
{tone}

WRITING RULES:
{style}

OUTPUT RULES:
{output}

EXAMPLES:
{examples}

EVENT:
{event}

TASK:
Generate LinkedIn posts according to all rules above.
"""


if __name__ == "__main__":
    # system components
    fs = FewShotPost()
    embedder = Embedder()
    builder = PromptBuilder("config/prompt_config.yaml")

    #1. Fake Interpreter output (structured event)
    interpreter_output = {
        "what_happened": "AI tools are transforming how developers build software",
        "why_global": "AI is reshaping global productivity",
        "why_nl": "Dutch startups are adopting AI rapidly",
        "why_kickstartai": "KickstartAI focuses on AI education and adoption",
        "stance": "AI should be adopted quickly in education",
        "arguments": [
            "Increases productivity",
            "Reduces development time",
            "Creates new opportunities"
        ]
    }

    #2. Convert event into structured text
    text = f"""
    What happened: {interpreter_output['what_happened']}
    Why global: {interpreter_output['why_global']}
    Why NL: {interpreter_output['why_nl']}
    Why KickstartAI: {interpreter_output['why_kickstartai']}
    Stance: {interpreter_output['stance']}
    Arguments: {', '.join(interpreter_output['arguments'])}
    """
    query_embedding = embedder.embed_text(text)

    # 4. Get similar posts
    few_shot_posts = fs.get_similar_posts(query_embedding, top_k=3)
    
    # 5. Buld prompt
    prompt = builder.build(interpreter_output, few_shot_posts)

    print(prompt)


# python3 -m src.Generator.prompt_builder









