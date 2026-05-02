import pandas as pd
from pandas import DataFrame
import json
from typing import Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from llm_clients import llm
import time
from tqdm import tqdm

delay = 2.5

def doc_to_json(doc_file: str, json_file: str) -> None:
    df : pd.DataFrame = pd.read_excel(doc_file, sheet_name="All posts", skiprows=1)

    # rename columns (for easier usage)
    df = df.rename(columns={"Post title": "text",
                            "Post link" : "url",
                            "Post type": "post_type",
                            "Created date": "created_date",
                            "Impressions": "impressions",
                            "Clicks": "clicks",
                            "Likes": "likes",
                            "Comments": "comments",
                            "Reposts": "reposts",
                            "Engagement rate": "engagement_rate"})
    
    df = df.fillna(0)
    records: list[dict] = df.to_dict(orient="records")

    with open(json_file, "w") as file:
        json.dump(records, file, indent=2)
    
    print(f"Saved processed data to {json_file}")

def process_posts(json_file: str, processed_json_file: str) -> None:
    enriched_posts = []
    with open(json_file, encoding="utf-8") as file:
        posts = json.load(file)
        for post in tqdm(posts[:3], desc="Processing LinkedIn posts.\n"):
            
            # Keep relevant fields 
            clean_post = {"text": post["text"],
                          "post_type": post["post_type"],
                          "created_date": post["created_date"]}
            
            # Add metadata (with LLM)
            metadata = extract_metadata(post["text"])
            
            clean_post["metadata"] = metadata
            enriched_posts.append(clean_post)

            time.sleep(delay)
    
    with open(processed_json_file, encoding="utf-8", mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

    print(f"Saved enriched dataset to {processed_json_file}")

def extract_metadata(post: str) -> dict[str, Any]:
    template = """
Extract a JSON object with exactly these keys:
- line_count (integer)
- language ("English" or "Dutch")
- tags (array of strings)

Rules:
1. tags MUST be only hashtags explicitly present in the text (e.g. #AI, #Data).
2. If there are NO hashtags, return an empty list [].
3. DO NOT infer or generate tags from the content.
4. DO NOT include keywords, topics, or names as tags unless they appear as hashtags.
5. Return ONLY valid JSON. No explanation.

Post:
{post}
"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke(input={"post": post})

    try: 
        json_parser = JsonOutputParser()
        json_res = json_parser.parse(response.content)
    except OutputParserException:
        raise OutputParserException("Failed to parse metadata JSON from LLM output.\n")

    return json_res


if __name__ == "__main__":
    raw_file = "data/KickstartAI LinkedIn post year to date.xlsx"
    json_file = "data/LinkedIn_data.json"
    processed_file = "data/LinkedIn_processed_data.json"

    doc_to_json(raw_file, json_file)
    process_posts(json_file, processed_file)