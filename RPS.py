import random

def player(prev_play, opponent_history=[]):
    # Save opponent history
    if prev_play:
        opponent_history.append(prev_play)

    # First move: random
    if len(opponent_history) == 0:
        return random.choice(["R", "P", "S"])

    # Helper: what beats what
    beats = {"R": "P", "P": "S", "S": "R"}

    # -------------------------------
    # Strategy 1: Counter repeating move (beats Quincy)
    if len(opponent_history) >= 3:
        if opponent_history[-1] == opponent_history[-2] == opponent_history[-3]:
            return beats[opponent_history[-1]]

    # -------------------------------
    # Strategy 2: Detect cycling (beats Abbey & Kris)
    if len(opponent_history) >= 6:
        last6 = opponent_history[-6:]
        if last6[:3] == last6[3:]:
            return beats[last6[0]]

    # -------------------------------
    # Strategy 3: Frequency analysis (beats Random)
    counts = {"R": 0, "P": 0, "S": 0}
    for move in opponent_history:
        counts[move] += 1

    most_common = max(counts, key=counts.get)
    return beats[most_common]
