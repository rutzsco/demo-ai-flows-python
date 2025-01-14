from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def status():
    return {"message": "Hello World"}
