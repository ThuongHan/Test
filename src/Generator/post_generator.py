from src.Generator.prompt_builder_v2 import PromptBuilder
from src.Generator.output_formatter import OutputFormatter
from src.utils.llm_clients import llm
from src.utils.few_shot import FewShotPost
from src.utils.embedder import Embedder
class PostGenerator:
    def __init__(self):
        self.llm = llm
        self.prompt_builder = PromptBuilder("config/prompt_config_v2.yaml")
        self.formatter = OutputFormatter()
        self.few_shot = FewShotPost()
        self.embedder = Embedder()

    def generate(self, interpreter_output: dict) -> dict:

        # 1. Convert event to text
        text = f"""
    What happened: {interpreter_output['what_happened']}
    Why global: {interpreter_output['why_global']}
    Why NL: {interpreter_output['why_nl']}
    Why KickstartAI: {interpreter_output['why_kickstartai']}
    Stance: {interpreter_output['stance']}
    Arguments: {', '.join(interpreter_output['arguments'])}
    """

        # 2. Embed query
        query_embedding = self.embedder.embed_text(text)

        # 3. Retrieve similar posts (few short exmaples)
        few_shot_posts = self.few_shot.get_similar_posts(
            query_embedding, top_k=3
        )

        # 4. Build prompt
        prompt = self.prompt_builder.build(
            interpreter_output,
            few_shot_posts
        )

        # 5. LLM call
        response = self.llm.invoke(prompt)

        formatted = self.formatter.format(
            response,
            few_shot_posts
        )

        return formatted
    

if __name__ == "__main__":
    
    interpreter_output = {
    "what_happened": (
        "ING COO Risk Leon Dusée shared lessons on AI adoption in large organizations, "
        "explaining why many AI pilots stall and how organizations can scale AI successfully."
    ),

    "why_global": (
        "Organizations worldwide are struggling with the same challenge: moving AI from "
        "isolated experiments into real operational impact. The bottleneck is often not "
        "the technology itself, but focus, governance, ownership, and organizational alignment."
    ),

    "why_nl": (
        "Dutch organizations across sectors like banking, aviation, retail, education, "
        "and healthcare are trying to accelerate AI adoption while avoiding fragmented "
        "pilot initiatives. Collaboration between organizations is becoming increasingly important "
        "for scaling AI responsibly and effectively in the Netherlands."
    ),

    "why_kickstartai": (
        "KickstartAI helps organizations share practical lessons about AI implementation, "
        "scaling, governance, and adoption. The collaboration between companies like ING, "
        "KLM, NS, and Ahold Delhaize aims to accelerate AI adoption in the Netherlands "
        "while also creating broader societal impact."
    ),

    "stance": (
        "AI adoption should be treated as an organizational transformation challenge, "
        "not just a technology experiment. Focus, leadership, and investing in people "
        "are just as important as the AI systems themselves."
    ),

    "arguments": [
        "AI pilots often fail because organizations think about scaling and ownership too late",
        "Organizations need to focus on a limited number of high-impact use cases instead of experimenting everywhere",
        "AI requires employees and leaders to develop critical thinking and adaptability",
        "The biggest AI opportunities often require redesigning entire processes, not just optimizing individual tasks",
        "Cross-organizational collaboration helps accelerate AI adoption and societal impact in the Netherlands",
        "AI should not only improve productivity, but also contribute to areas like financial health, education, and healthcare"
    ],

    "source": "https://lnkd.in/eU8jqvT9"
}

    generator: PostGenerator = PostGenerator()
    result = generator.generate(interpreter_output)

    print("\nGENERATED POSTS:\n")
    print(result["posts"])
    print("-"*60)
    
    print("\nUSED FEW-SHOT POSTS:\n")
    for i, post in enumerate(result["few_shot_examples"], 1):
        print(f"\n EXAMPLE {i}")
        print(post["text"])
# python3 -m src.Generator.post_generator






