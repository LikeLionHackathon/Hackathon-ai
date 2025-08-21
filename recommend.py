from pydantic import BaseModel
from typing import List, Optional

class Exhibition(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    posterImageUrl: Optional[str] = None
    artworkImages: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    recommendationReason: Optional[str] = None

import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

from typing import List, Dict, Optional

def analyze_exhibition_with_images_and_fs(
    prompt: str,
    image_blobs: List[bytes],                # 이미지 바이트 리스트
    vector_store_id: Optional[str] = None,
) -> Exhibition:
    # 벡터 스토어 ID 확정 (인자 > 환경변수)
    vs_id = vector_store_id
    if not vs_id:
        raise ValueError(
            "VECTOR_STORE_ID is not set. Pass vector_store_id to the function or set it in your environment/.env."
        )

    # content_blocks 구성: 텍스트 + 여러 이미지(바이트)
    content_blocks = [{"type": "input_text", "text": prompt}]
    for data in image_blobs:
        content_blocks.append({"type": "input_image", "image_data": data})

    # File Search 도구를 첨부하고, 구조화 파싱으로 Exhibition 받아오기
    resp = client.responses.parse(
        model="gpt-4.1-mini",
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vs_id]
        }],
        input=[{
            "role": "user",
            "content": content_blocks
        }],
        text_format=Exhibition
    )
    return resp.output_parsed

from typing import Optional

def process_images(images_info: list[dict], text: Optional[str] = None, vector_store_id: Optional[str] = None) -> Exhibition:
    """
    images_info의 각 항목은 다음 키를 가질 것을 권장:
    {
      "filename": str,
      "content_type": str,
      "size": int,
      "bytes": bytes    # ★ 여기 필수 (OpenAI에 image_data로 보낼 바이트)
    }
    """
    # 1) 안전하게 이미지 바이트만 추출
    image_blobs: List[bytes] = []
    for info in images_info:
        data = info.get("bytes")
        if not data:
            # bytes가 없다면 스킵하거나 에러 처리 선택
            # 여기선 스킵
            continue
        image_blobs.append(data)

    # 2) 프롬프트(텍스트) 구성
    prompt = f"{text} 이미지와 내 전시 지식에서 가장 잘 맞는 전시를 추천해줘. 이유와 관련 태그도 포함." if text else "이미지와 내 전시 지식에서 가장 잘 맞는 전시를 추천해줘. 이유와 관련 태그도 포함."
    # 3) File Search + 이미지 비전 호출
    exhibition = analyze_exhibition_with_images_and_fs(
        prompt=prompt,
        image_blobs=image_blobs,
        vector_store_id=vector_store_id,
    )
    return exhibition