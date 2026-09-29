def calculate_risk_score(row, threat_probability):
    """
    Convert model probability and behavioral evidence
    into a 0-100 ARGUS risk score.
    """

    score = threat_probability * 60

    # Process indicators
    if row.get("powershell_count", 0) > 0:
        score += 8

    if row.get("cmd_count", 0) > 0:
        score += 5

    if row.get("suspicious_parent_child_count", 0) > 0:
        score += 10

    # File indicators
    if row.get("file_write_velocity", 0) > 1:
        score += 5

    if row.get("high_entropy_write_count", 0) > 5:
        score += 8

    if row.get("entropy_delta_mean", 0) > 1:
        score += 8

    # Registry indicators
    if row.get("persistence_key_count", 0) > 0:
        score += 10

    # Network indicators
    if row.get("network_connection_count", 0) > 5:
        score += 3

    # Authentication indicators
    if row.get("failed_login_count", 0) > 3:
        score += 5

    score = max(0, min(100, score))

    if score >= 75:
        severity = "CRITICAL"
    elif score >= 50:
        severity = "HIGH"
    elif score >= 25:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return round(score, 2), severity