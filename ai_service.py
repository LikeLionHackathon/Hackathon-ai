from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional
from typing import Any
from fastapi import HTTPException



# .env 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 클라이언트 생성
client = OpenAI(api_key=api_key)
# Python
def ensure_vector_store_exists(vector_store_id: str):
    """Retrieve the vector store to validate the ID; raise a helpful error if invalid."""
    try:
        vs = client.vector_stores.retrieve(vector_store_id)
        # Optional: print minimal info for debugging
        print("[vector_store] ok:", getattr(vs, "id", vector_store_id))
        return vs
    except Exception as e:
        # Surface a clear message up to the API layer
        raise HTTPException(status_code=500, detail=f"Invalid VECTOR_STORE_ID '{vector_store_id}': {e}")

class Exhibition(BaseModel):
    tags: list[str]


def analyze_exhibition(
    prompt: str, 
    image_urls: list[str], 
    vector_store_id: str
) -> Exhibition:
    
    content_blocks = [{"type": "input_text", "text": prompt}]

    # 이미지 여러 개 추가
    for url in image_urls:
        content_blocks.append({"type": "input_image", "image_url": url})

    response = client.responses.parse(
        model="gpt-4.1-mini",
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id]
        }],
        input=[{
            "role": "user",
            "content": content_blocks
        }],
        text_format=Exhibition
    )

    return response.output_parsed






