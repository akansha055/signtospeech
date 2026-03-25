import os

gestures = ["hello","thankyou","yes","no","please",
            "stop","water","help","sorry","come"]

for g in gestures:
    os.makedirs(f"dataset/{g}", exist_ok=True)