from typing import Optional

def map_leg_notes(score: float) -> str:
    """
    Maps a per-leg score to a clinical note.
    
    Logic:
    > 7      : "Horse seems HEALTHY!"
    7 -> 5   : "No immediate concern on this leg"
    4 -> 3   : "Need a regular checkup on this leg"
    < 3      : "Immediate checkup required"
    """
    if score > 7:
        return "Horse seems HEALTHY!"
    elif score >= 5:
        return "No immediate concern on this leg"
    elif score >= 3:
        return "Need a regular checkup on this leg"
    else:
        return "Immediate checkup required"
