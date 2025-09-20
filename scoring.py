def final_score(hard_score, semantic_score, hard_weight=0.6, semantic_weight=0.4):
    return hard_score * hard_weight + semantic_score * semantic_weight

def verdict(score):
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    else:
        return "Low"