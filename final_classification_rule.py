class AdvancedResumeClassifier:
    def __init__(self):
        # Weight configuration
        self.jd_weight = 0.6
        self.resume_weight = 0.4

        # Hard thresholds (strict filters)
        self.min_jd_required = 40
        self.min_resume_required = 35

        # Decision thresholds
        self.shortlist_threshold = 75
        self.review_threshold = 55

    def classify(self, jd_score, resume_score):
        """
        Advanced rule-based classification
        """

        # --- VALIDATION ---
        jd_score = max(0, min(100, jd_score))
        resume_score = max(0, min(100, resume_score))

        # --- HARD FILTERS ---
        if jd_score < self.min_jd_required:
            return {
                "decision": "Reject",
                "final_score": 0,
                "reason": "Very low job relevance (JD match too low)"
            }

        if resume_score < self.min_resume_required:
            return {
                "decision": "Reject",
                "final_score": 0,
                "reason": "Poor resume quality"
            }

        # --- WEIGHTED SCORING ---
        final_score = (
            self.jd_weight * jd_score +
            self.resume_weight * resume_score
        )

        # --- BONUS LOGIC (smart adjustment) ---
        bonus = 0

        # High JD match bonus
        if jd_score > 85:
            bonus += 3

        # Strong resume bonus
        if resume_score > 80:
            bonus += 2

        # Penalize imbalance
        if abs(jd_score - resume_score) > 30:
            bonus -= 5  # mismatch penalty

        final_score += bonus
        final_score = max(0, min(100, final_score))

        # --- DECISION ---
        if final_score >= self.shortlist_threshold:
            decision = "Shortlist"
            reason = "Strong relevance and high-quality profile"

        elif final_score >= self.review_threshold:
            decision = "Review"
            reason = "Moderate match, requires manual evaluation"

        else:
            decision = "Reject"
            reason = "Not sufficiently aligned with job or quality standards"

        return {
            "decision": decision,
            "final_score": round(final_score, 2),
            "jd_score": jd_score,
            "resume_score": resume_score,
            "reason": reason
        }