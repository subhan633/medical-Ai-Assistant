from fastapi import APIRouter, UploadFile, File
from typing import List
from modules.load_vectorstore import load_vectorestore
from fastapi.responses import JSONResponse
from logger import logger

router=APIRouter()

@router.post("/upload_pdf/")
async def upload_pdf(files:List[UploadFile] = File(...)):
    try:
        logger.info("Received uploaded files")
        # To:
        await load_vectorestore(files)
        logger.info("Document added to vectorestore")
        return {"message": "Files processed and vectorestore updated"}
    except Exception as e:
        logger.exception("Error during Uploading PDF")
        return JSONResponse(status_code=500, content={"error":str(e)})