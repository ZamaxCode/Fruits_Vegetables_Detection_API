from fastapi import APIRouter,UploadFile
from ..process.process_image import get_results


router = APIRouter()

@router.post("/inference/") 
async def inference_fv(image: UploadFile):
    image_data = await image.read()
    response = get_results(image_data)
    return response
