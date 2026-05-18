import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 0. Configuration & Models
# ============================================================

SCANNER_OUTPUT_PATH = os.getenv("SCANNER_OUTPUT_PATH", "config/scanner_output.json")
BELIEF_REPOSITORY_PATH = os.getenv("BELIEF_REPOSITORY_PATH", "config/belief_repository.json")
INTERPRETER_OUTPUT_PATH = os.getenv("INTERPRETER_OUTPUT_PATH", "interpreter_to_generator.json")

REQUIRED_CONTRACT_KEYS = [
    "What happened",
    "Why does it matter (globally and NL)",
    "Why does it matter for KickstartAI",
    "Key stance / opinion",
    "Supporting arguments",
]


@dataclass
class Article:
    title: str
    full_text: str
    raw: Dict[str, Any]


@dataclass
class RAGResult:
    belief: str
    score: float


# ============================================================
# 0. OpenAI Client Factory (secure)
# ============================================================

def create_client() -> OpenAI:
    """
    Create an OpenAI client using UVA_API_TOKEN from environment.
    Fail-fast if token is missing.
    """
    token = os.getenv("UVA_API_TOKEN")
    if not token:
        raise RuntimeError(
            "UVA_API_TOKEN is not set. "
            "Export it as an environment variable instead of hardcoding it."
        )

    client = OpenAI(
        api_key=token,
        base_url="https://llmproxy.uva.nl/v1/",
    )
    print("UvA LLM Proxy Client created")
    return client
client = create_client()

def step1_ingest_data():
    """
    旧版本的 Step 1：返回 (article_dict, beliefs)

    - ingest_data() 返回的是 Article 对象 + belief_texts
    - 我们把 Article.raw 还原成 dict，belief_texts 原样传下去
    """
    article, belief_texts = ingest_data()  # 调用你现在的工业版 ingest_data
    return article.raw, belief_texts
  # ============================================================
# STEP 2: RAG（可扩展接口，当前默认 TF-IDF）
# ============================================================

class BaseRetriever(ABC):
    """
    统一的 RAG 接口，方便以后换成 Embedding 检索。
    """

    @abstractmethod
    def retrieve(self, article: Article, beliefs: List[str]) -> RAGResult:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    """
    现在用的 TF-IDF baseline。将来可以并存 EmbeddingRetriever。
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)

    def retrieve(self, article: Article, beliefs: List[str]) -> RAGResult:
        print("[2/4] Applying RAG engine (TF-IDF)")

        news_content = f"{article.title}\n{article.full_text}"
        belief_texts = [
            b["belief"] if isinstance(b, dict) and "belief" in b else str(b)
            for b in beliefs
        ]

        if not belief_texts:
            raise ValueError("No 'belief' texts provided to RAG")

        all_texts = [news_content] + belief_texts

        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        best_idx = int(sims.argmax())
        best_belief = belief_texts[best_idx]
        highest_score = float(sims[best_idx])

        print("\n" + "=" * 60)
        print(f"news title: {article.title}")
        print(f"RAG retrieving succeed (TF-IDF, score: {highest_score:.4f})")
        print(f"KickstartAI belief matched:\n{best_belief}")
        print("=" * 60 + "\n")

        return RAGResult(belief=best_belief, score=highest_score)


# 如果你想以后用 Embedding-RAG，可以在这里加一个 EmbeddingRetriever，
# 目前先不动，保持简单：所有调用默认用 TfidfRetriever。


# 原来的 step2_rag_retrieval：改成简单包装，保持兼容老调用习惯
def step2_rag_retrieval(article_dict: Dict[str, Any], beliefs: List[Dict[str, Any]]) -> RAGResult:
    """
    兼容老接口的包装：
      - 输入还是原来的 article dict + belief list
      - 内部转成 Article，再交给 Retriever
    """
    article = Article(
        title=article_dict.get("name") or article_dict.get("title") or "Unknown Title",
        full_text=article_dict.get("full_text") or article_dict.get("content") or "",
        raw=article_dict,
    )
    retriever = TfidfRetriever()
    return retriever.retrieve(article, beliefs)
# ============================================================
# STEP 3: Prompt（支持 CoT / Flat 两种模式）
# ============================================================

ReasoningMode = Literal["cot", "flat"]


def build_cot_prompt(article: Article, retrieved_belief: str) -> str:
    print("[3/4] Building CoT prompt...")

    news_excerpt = article.full_text[:1500]

    prompt = f"""
You are the Chief Strategic Interpreter for KickstartAI.
Your task is to analyze external industry news and determine our strategic stance
based strictly on our retrieved internal guidelines.

[EXTERNAL FACT: NEWS ARTICLE]
Title: {article.title}
Content: {news_excerpt}

[INTERNAL BRAIN: RETRIEVED KICKSTARTAI BELIEF]
Strictly align your analysis with this core mission/belief:
\"\"\"{retrieved_belief}\"\"\"

[CHAIN OF THOUGHT REASONING]
Before outputting the final JSON, mentally process the information through these 3 steps:
- Step 1 (Fact Extraction): What exactly happened in the news, and what is its global and Dutch (NL) macro impact?
- Step 2 (Strategic Alignment): How does this specific event align with or challenge our retrieved KickstartAI belief?
- Step 3 (Stance Synthesis): Based on the alignment, what should our official stance be, and what are 3 key arguments to support it?

[STRICT OUTPUT FORMAT]
You must output ONLY a valid JSON object matching EXACTLY the keys below.
Do not include markdown formatting like ```json or natural language commentary.

{{
    "What happened": "Brief objective summary of the event",
    "Why does it matter (globally and NL)": "Macro industry impact and local Dutch context",
    "Why does it matter for KickstartAI": "Direct connection to the retrieved belief",
    "Key stance / opinion": "Supportive / Critical / Neutral / Cautious",
    "Supporting arguments": [
        "Argument 1",
        "Argument 2",
        "Argument 3"
    ]
}}
""".strip()

    print("CoT prompt assembled")
    return prompt


def build_flat_prompt(article: Article, retrieved_belief: str) -> str:
    print("[3/4] Building FLAT prompt (no explicit CoT)...")

    news_excerpt = article.full_text[:1500]

    prompt = f"""
You are the Chief Strategic Interpreter for KickstartAI.
Analyze the following news and directly provide a concise strategic assessment
for KickstartAI, strictly aligned with our internal belief.

[NEWS ARTICLE]
Title: {article.title}
Content: {news_excerpt}

[INTERNAL BELIEF]
\"\"\"{retrieved_belief}\"\"\"

[OUTPUT REQUIREMENT]
Directly output ONLY a JSON object with the following keys, no explanations,
no reasoning steps, no markdown:

{{
    "What happened": "Brief objective summary of the event",
    "Why does it matter (globally and NL)": "Macro industry impact and local Dutch context",
    "Why does it matter for KickstartAI": "Direct connection to the retrieved belief",
    "Key stance / opinion": "Supportive / Critical / Neutral / Cautious",
    "Supporting arguments": [
        "Argument 1",
        "Argument 2",
        "Argument 3"
    ]
}}
""".strip()

    print("Flat prompt assembled")
    return prompt


def step3_build_prompt(
    article_dict: Dict[str, Any],
    retrieved_belief: str,
    mode: ReasoningMode = "cot",
) -> str:
    """
    兼容老接口：输入还是 article dict，内部转 Article，
    然后根据 mode 选择 CoT / Flat。
    """
    article = Article(
        title=article_dict.get("name") or article_dict.get("title") or "Unknown Title",
        full_text=article_dict.get("full_text") or article_dict.get("content") or "",
        raw=article_dict,
    )

    if mode == "cot":
        return build_cot_prompt(article, retrieved_belief)
    elif mode == "flat":
        return build_flat_prompt(article, retrieved_belief)
    else:
        raise ValueError(f"Unknown reasoning mode: {mode}")

# ==========================================
# STEP 4: LLM Output & Contract Validation
# ==========================================
def step4_execute_llm(prompt_string: str):
    print("\n[4/4] Invoking UvA LLM Proxy (JSON Mode)...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a KickstartAI strategic analyst. "
                        "You MUST output strictly in valid JSON format "
                        "matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt_string},
            ],
            temperature=0.3,
        )

        raw_content = response.choices[0].message.content
        final_json = json.loads(raw_content)

        # ---- Data Contract Check ----
        required_keys = [
            "What happened",
            "Why does it matter (globally and NL)",
            "Why does it matter for KickstartAI",
            "Key stance / opinion",
            "Supporting arguments",
        ]

        missing_keys = [key for key in required_keys if key not in final_json]
        if missing_keys:
            raise ValueError(f"Contract Violation! LLM missed these keys: {missing_keys}")

        if not isinstance(final_json.get("Supporting arguments"), list):
            raise ValueError("Contract Violation! 'Supporting arguments' must be a list")

        # Save JSON File
        with open("interpreter_to_generator.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)

        print("\n✅ [SUCCESS] Contract Verified. Final Strategic Output:\n")
        print(json.dumps(final_json, indent=4, ensure_ascii=False))

        return final_json

    except json.JSONDecodeError:
        print(
            f"\n❌ [ERROR] JSON Parsing Failed. "
            f"The proxy might have stripped the json_object parameter. "
            f"Raw Output: {raw_content}"
        )
    except Exception as e:
        print(f"\n❌ [ERROR] API or Execution failed: {e}")

# ============================================================
# STEP 4: Operator（主入口）
# ============================================================

def run_interpreter_from_files(
    reasoning_mode: ReasoningMode = "cot",
):
    """
    用文件作为输入的完整流水线：
      1. Step1 ingest
      2. Step2 RAG (TF-IDF)
      3. Step3 Prompt (CoT / Flat)
      4. Step4 LLM 调用 + 合同校验
    """
    print("=== KickstartAI Interpreter ===")

    # Step 1: ingest
    article_dict, beliefs = step1_ingest_data()

    # Step 2: RAG（返回 RAGResult）
    rag_result = step2_rag_retrieval(article_dict, beliefs)

    # Step 3: Prompt（根据 mode 选择 CoT / Flat）
    final_prompt = step3_build_prompt(
        article_dict,
        rag_result.belief,
        mode=reasoning_mode,
    )

    # Step 4: LLM
    result = step4_execute_llm(final_prompt)
    return result


if __name__ == "__main__":
    try:
        # 默认用 CoT；想做消融实验可以改成 "flat"
        run_interpreter_from_files(reasoning_mode="cot")
    except Exception as exc:
        print(f"\nFATAL: Interpreter failed: {exc}")
        raise

