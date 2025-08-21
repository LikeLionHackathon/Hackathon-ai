from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
from fastapi.encoders import jsonable_encoder
from ai_service import analyze_exhibition
from datetime import date
from typing import List, Optional
from fastapi import File, UploadFile,Form, HTTPException
from recommend import ask_with_images_via_files
from upload import save_exhibition_to_vector_store
import base64

load_dotenv()



VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID", "vs_68a06b40e62481918fce07b01bb734cc")
EXHIBITION_STORE_ID=os.getenv("EXHIBITION_STORE_ID","vs_68a5414513b081919a2c0980a2d4437b")
app = FastAPI()

class AiTagRequest(BaseModel):
    id:int
    title: str
    startDate: date
    endDate: date
    teamName: str
    location: str
    description: str
    posterImageUrl: str
    artworkImages: List[str]


@app.post("/tags")
def generate_tags(req: AiTagRequest):
    # 1) AI 호출
    print("여기까지 ok")
    result = analyze_exhibition(

        prompt=(
            "다음은 전시 설명과 전시 포스터, 전시 작품 이미지야 "
            "전시에 어울리는 태그 3개 추천해줘\n"
            f"설명 : {req.description}, 태그는 내가 미리 첨부했던 tag.txt 파일에서만 추천"
        ),
        image_urls=[req.posterImageUrl] + req.artworkImages,
        vector_store_id=VECTOR_STORE_ID
    )
    print(result)
    
    # 2) 벡터 스토어에 '요청+응답' 합본 문서 저장
    upload_info = save_exhibition_to_vector_store(
        req=req,
        tags=result.tags,
        vector_store_id=EXHIBITION_STORE_ID
    )
    print("완료")

    # 3) 클라이언트로 응답 (태그 + 업로드 메타 반환)
    return {
        "tags": result.tags,
    }


class AiTagResponse(BaseModel):
    tags: List[str]
    message: Optional[str] = None

# @app.post("/recommend", response_model=List[Exhibition])  # ← 리스트로 변경
# async def recommend(
#     text: Optional[str] = Form(None),
#     artworkImages: Optional[List[UploadFile]] = File(None)
# ):
#     images_info = []
#     if artworkImages:
#         for f in artworkImages:
#             if not f.content_type or not f.content_type.startswith("image/"):
#                 raise HTTPException(status_code=415,
#                                     detail=f"Unsupported type: {f.content_type}")
#             data = await f.read()
#             images_info.append({
#                 "filename": f.filename,
#                 "content_type": f.content_type,
#                 "size": len(data),
#                 "_bytes": data,  # 내부 전용 키(직렬화 방지)
#             })

#     exhibitions = process_images(images_info, text, EXHIBITION_STORE_ID)  # List[Exhibition]
#     return jsonable_encoder(exhibitions)

@app.post("/recommend")
async def recommend(
    text: Optional[str] = Form(None),
    artworkImages: Optional[List[UploadFile]] = File(None)
):

    result = ask_with_images_via_files(text or "전시 추천해줘", artworkImages)
    return {"recommendations": result}