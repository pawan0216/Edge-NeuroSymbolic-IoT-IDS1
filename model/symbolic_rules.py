def symbolic_rules(features):
    rules_triggered = []

    if features.mean() > 0.5:
        rules_triggered.append("High traffic anomaly")

    if features.max() > 2.0:
        rules_triggered.append("Burst behavior detected")

    return rules_triggered
