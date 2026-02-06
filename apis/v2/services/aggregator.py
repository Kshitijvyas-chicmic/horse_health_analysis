from typing import List, Optional

def aggregate_scan(leg_scores: List[float]) -> dict:
    """
    Aggregates per-leg scores into an overall result using priority-based logic.
    
    Priority Logic:
    1. If ANY leg score == 1 → CRITICAL
    2. Else if ANY leg score is 2-5 → CONCERN
    3. Else if ALL legs are 9-10 → OPTIMAL
    4. Else (all legs 6-8) → ACCEPTABLE
    
    Returns:
    {
        "scanScore": float,  # Average of all legs
        "notes": str         # Combined assessment + recommendation
    }
    """
    if not leg_scores:
        return {
            "scanScore": None,
            "notes": "No legs analyzed."
        }
    
    # Calculate statistics
    min_score = min(leg_scores)
    avg_score = sum(leg_scores) / len(leg_scores)
    avg_score = round(avg_score, 1)
    
    # Priority-based decision tree
    if min_score == 1:
        # CRITICAL: At least one leg has severe misalignment
        notes = "Severe issue detected in at least one leg. Immediate veterinary and farrier evaluation recommended."
    
    elif min_score <= 5:
        # CONCERN: At least one leg has poor alignment
        notes = "Overall leg health is fair, but one or more legs show concerning misalignment. Corrective hoof care is advised. Veterinary review recommended if discomfort is present."
    
    elif min_score >= 9:
        # OPTIMAL: All legs have excellent alignment
        notes = "Excellent overall hoof–pastern alignment across all legs. No action needed beyond routine maintenance."
    
    else:
        # ACCEPTABLE: All legs are in the 6-8 range
        notes = "Overall leg alignment is acceptable. Continue regular hoof care and routine monitoring."
    
    return {
        "scanScore": avg_score,
        "notes": notes
    }
