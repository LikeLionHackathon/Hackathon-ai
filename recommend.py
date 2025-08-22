# recommend.py
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi import UploadFile, HTTPException
import base64, os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
EXHBITION_STORE_ID = os.getenv("EXHIBITION_STORE_ID")


class Exhibition(BaseModel):
    id: Optional[int] = None
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    posterImageUrl: Optional[str] = None
    artworkImages: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    recommendationReason: Optional[str] = None

# ✅ 리스트 래퍼
class ExhibitionList(BaseModel):
    items: List[Exhibition]





# def analyze_exhibition_with_images_and_fs(
#     prompt: str,
#     image_blobs: List[bytes],
#     vector_store_id: Optional[str] = None,
# ) -> List[Exhibition]:  # ← 리스트로 변경
#     if not vector_store_id:
#         raise ValueError("VECTOR_STORE_ID is not set.")

#     content_blocks: List[Dict] = [{"type": "input_text", "text": prompt}]
#     for data in image_blobs:
#         b64 = base64.b64encode(data).decode("ascii")
#         content_blocks.append({
#             "type": "input_image",
#             "image_data": {"data": b64, "mime_type": "image/png"}
#         })

#     req = {
#         "model": "gpt-4.1-mini",
#         "tools": [{"type": "file_search",
#                    "vector_store_ids": [vector_store_id]}],
#         "input": [{"role": "user", "content": content_blocks}],
#         "text_format": ExhibitionList,   # ← 다중 항목 파싱
#     }
#     _assert_no_bytes(req)

#     resp = client.responses.parse(**req)
#     return resp.output_parsed.items  # List[Exhibition]


# def process_images(
#     images_info: List[dict],
#     text: Optional[str] = None,
#     vector_store_id: Optional[str] = None
# ) -> List[Exhibition]:  # ← 리스트로 변경
#     image_blobs: List[bytes] = []
#     for info in images_info or []:
#         data = info.get("_bytes") or info.get("bytes")
#         if data:
#             image_blobs.append(data)

#     prompt = (
#         f"{text} 이미지를 참고해 내 전시 지식에서 **가장 잘 맞는 전시 여러 개**를 추천해줘. "
#         "각 항목마다 이유와 관련 태그 포함."
#         if text else
#         "이미지를 참고해 내 전시 지식에서 **가장 잘 맞는 전시 여러 개**를 추천해줘. 각 항목마다 이유와 관련 태그 포함."
#     )

#     return analyze_exhibition_with_images_and_fs(
#         prompt=prompt,
#         image_blobs=image_blobs,
#         vector_store_id=vector_store_id,
#     )

def ask_with_images_via_files(prompt: str, images: List[UploadFile]):
    file_ids: List[str] = []
    for f in images or []:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail=f"Unsupported type: {f.content_type}")

        data = f.file.read()
        uploaded = client.files.create(
            file=(f.filename or "image", data, f.content_type),
            purpose="user_data",
        )
        file_ids.append(uploaded.id)

    content = [{
        "type": "input_text",
        "text": (
            "prompt : " + prompt + "\n"
            "사용자가 전송한 이미지 + prompt를 바탕으로 업로드한 파일/지식에서 "
            "맞춤 전시 여러 개를 추천하고, 각 추천 이유도 생성해줘."
        )
    }]

    for fid in file_ids:
        content.append({
            "type": "input_image",
            "file_id": fid   # ✅ 수정 포인트
        })

    resp = client.responses.parse(
        model="gpt-4o-mini",
        tools=[{
            "type": "file_search",
            "vector_store_ids": [EXHBITION_STORE_ID]
        }],
        input=[{"role": "user", "content": content}],
        text_format=ExhibitionList,
    )
    return resp.output_parsed.items