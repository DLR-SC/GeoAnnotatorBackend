from fastapi import APIRouter, HTTPException
from utils.baseModels import FeedbackRequest
from activeLearning.activeLearningFunctions import *

from dotenv import load_dotenv
import os

load_dotenv()
FEEDBACK_DIR_PATH=os.getenv("FEEDBACK_DIR_PATH")

router = APIRouter()

@router.get("/feedback")
async def countFeedback(instance_name: str):
    try:
        feedback_data = await load_feedback(instance_name, FEEDBACK_DIR_PATH)

        return { "feedback_count": len(feedback_data) }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Counting feedback failed! {str(e)}")

@router.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        os.makedirs(FEEDBACK_DIR_PATH, exist_ok=True)

        # Save feedback
        await store_feedback(request, FEEDBACK_DIR_PATH)
        # Check, if Retrain-Job needs to be initiated
        await check_feedback_threshold(request.provider, FEEDBACK_DIR_PATH)

        return { "message": "Feedback processed successfully." }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))