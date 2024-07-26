from fastapi import APIRouter,UploadFile, HTTPException
from ..process.process_image import get_results


router = APIRouter()

@router.post("/inference/") 
async def inference_fv(image: UploadFile):
    try:
        image_data = await image.read()
    except:
        raise HTTPException(status_code=400, detail="Can't read image.")
    try:
        response = get_results(image_data)
    except:
        response = {'detections': []}
    return response
