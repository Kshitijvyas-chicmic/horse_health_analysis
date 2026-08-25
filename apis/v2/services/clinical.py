from typing import Optional

def map_condition(score: Optional[float]) -> Optional[str]:
    """
    Maps a per-leg score to a clinical condition.
    
    Score Ranges:
    - 9-10: Optimal Alignment
    - 6-8:  Acceptable Alignment
    - 2-5:  Poor Alignment
    - 1:    Critical Misalignment
    """
    if score is None:
        return None
    
    if score >= 9:
        return "Optimal Alignment"
    elif score >= 6:
        return "Acceptable Alignment"
    elif score >= 2:
        return "Poor Alignment"
    else:  # score == 1
        return "Critical Misalignment"


def map_clinical_notes(score: Optional[float]) -> Optional[str]:
    """
    Maps a per-leg score to detailed clinical notes.
    
    Uses exact text as specified in requirements.
    """
    if score is None:
        return None
    
    if score >= 9:
        return "Hoof and pastern angles well aligned."
    elif score >= 6:
        return "Minor deviation. Generally functional mechanics."
    elif score >= 2:
        return "Noticeable angle deviation. Uneven load on limb."
    else:  # score == 1
        return "Severe hoof–pastern angle mismatch. High limb stress."


def map_recommendation(score: Optional[float]) -> Optional[str]:
    """
    Maps a per-leg score to clinical recommendations.
    
    Uses exact text as specified in requirements.
    """
    if score is None:
        return None
    
    if score >= 9:
        return "No action needed. Continue current hoof care routine."
    elif score >= 6:
        return "Maintain regular trimming. Recheck after next shoeing cycle."
    elif score >= 2:
        return "Schedule corrective trimming soon. Monitor closely for discomfort."
    else:  # score == 1
        return "Urgent farrier and veterinary evaluation advised. Limit work until corrected."

import os
import json
import logging
import asyncio

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

async def generate_clinical_insights(metrics: dict) -> dict:
    """
    Dynamically generates clinical notes and recommendations using Gemini,
    based on the full hoof metrics dictionary.
    Falls back to hardcoded mappings on failure.
    """
    score = metrics.get("score")
    fallback_notes = map_clinical_notes(score)
    fallback_rec = map_recommendation(score)
    
    if genai is None:
        logging.warning("google.generativeai not installed. Using fallback clinical notes.")
        return {"notes": fallback_notes, "recommendation": fallback_rec}
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.warning("GEMINI_API_KEY not found. Using fallback clinical notes.")
        return {"notes": fallback_notes, "recommendation": fallback_rec}
        
    try:
        models_to_try = [
            'gemini-3.6-flash',
            'gemini-3.5-flash-lite',
            'gemini-flash-latest'      # Catch-all safe fallback
        ]
        
        client = genai.Client(api_key=api_key)
        result = None
        for model_name in models_to_try:
            try:


                
                prompt = (
                    "You are an expert veterinary assistant and farrier. I am providing you with the "
                    "Hoof-Pastern Axis (HPA) measurement metrics for a horse's leg.\n"
                    f"Metrics: {json.dumps(metrics, indent=2)}\n\n"
                    "Based on these metrics, provide 'notes' (a horse-owner friendly explanation of the current state). "
                    "Keep the language professional, encouraging, and easy to understand for a horse owner.\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- Your response MUST be extremely concise, strictly limited to a maximum of 17-20 words, and no more than 2 short sentences.\n"
                    "- Do NOT use bullet points or lists in your response.\n"
                    "- Do NOT include any actionable advice, treatment plans, or recommendations in this note.\n"
                    "- Do NOT mention the numerical angles, score, or deviation in your response, as these are already displayed on the screen.\n"
                    "Return ONLY a JSON object with one key: 'notes'."
                )
                
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                result = json.loads(response.text)
                logging.info(f"Successfully generated clinical notes using {model_name}")
                break
            except Exception as e:
                logging.warning(f"Failed to generate clinical notes with {model_name}: {e}")
                continue
                
        if not result:
            logging.error("All Gemini models failed to generate clinical notes.")
            return {"notes": fallback_notes, "recommendation": fallback_rec}
            
        # We explicitly return None for recommendation as it will be generated at the top level
        return {
            "notes": result.get("notes", fallback_notes),
            "recommendation": None
        }
    except Exception as e:
        logging.error(f"Error generating clinical notes with Gemini: {e}")
        return {"notes": fallback_notes, "recommendation": fallback_rec}

async def generate_overall_recommendations(all_metrics: list) -> Optional[str]:
    """
    Generates an overall clinical recommendation for the entire horse based on metrics from all legs.
    Returns a string formatted as short bullet points.
    """
    if genai is None or not all_metrics:
        return None
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    try:
        client = genai.Client(api_key=api_key)
        models_to_try = [
            'gemini-3.6-flash',
            'gemini-3.5-flash-lite',
            'gemini-flash-latest'
        ]
        
        prompt = (
            "You are an expert veterinary assistant and farrier. I am providing you with the "
            "Hoof-Pastern Axis (HPA) measurement metrics for a horse's legs.\n"
            f"Metrics:\n{json.dumps(all_metrics, indent=2)}\n\n"
            "Based on these metrics, generate an overall clinical recommendation for the horse. "
            "CRITICAL INSTRUCTIONS:\n"
            "- The recommendation MUST be formatted as a list of short statements, separated by new lines. Do NOT use bullet points or the '•' character.\n"
            "- Each statement must be very short and concise (e.g. 'Monitor white line separation closely.', 'Consult with a farrier for trimming adjustments.').\n"
            "- Do NOT write a paragraph. Do NOT include numerical angles.\n"
            "Return ONLY a JSON object with one key: 'recommendation'."
        )
        
        for model_name in models_to_try:
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                result = json.loads(response.text)
                return result.get("recommendation")
            except Exception as e:
                logging.warning(f"Failed to generate overall recommendation with {model_name}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Error generating overall recommendation with Gemini: {e}")
        
    return None
