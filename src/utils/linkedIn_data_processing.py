import pandas as pd
from pandas import DataFrame
import json
from typing import Any
from llm_clients import llm
import time
from tqdm import tqdm
import re


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
        for post in tqdm(posts, desc="Processing LinkedIn posts.\n"):
            # Keep relevant fields 
            clean_post = {"text": post["text"],
                          "post_type": post["post_type"],
                          "created_date": post["created_date"],
                          "metadata": {"line_count": post["text"].count("\n") + 1,
                                       "tags": re.findall(r"#\w+", post["text"])
                                       }
                          }

            enriched_posts.append(clean_post)
    
    with open(processed_json_file, encoding="utf-8", mode="w") as outfile:
        json.dump(enriched_posts, outfile, indent=4)

    print(f"Saved enriched dataset to {processed_json_file}")



if __name__ == "__main__":
    raw_file = "data/KickstartAI LinkedIn post year to date.xlsx"
    json_file = "data/LinkedIn_data.json"
    processed_file = "data/LinkedIn_processed_data.json"

    doc_to_json(raw_file, json_file)
    process_posts(json_file, processed_file)