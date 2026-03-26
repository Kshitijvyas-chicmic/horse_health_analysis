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
