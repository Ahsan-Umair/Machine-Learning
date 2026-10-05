import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student_performance.csv")

# Sample only for faster plotting
sample = df.sample(5000, random_state=42)

# ==========================================
# Study Hours vs Total Score
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(
    sample["weekly_self_study_hours"],
    sample["total_score"]
)

plt.title("Study Hours vs Total Score")
plt.xlabel("Weekly Self Study Hours")
plt.ylabel("Total Score")

plt.show()

# ==========================================
# Attendance vs Total Score
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(
    sample["attendance_percentage"],
    sample["total_score"]
)

plt.title("Attendance vs Total Score")
plt.xlabel("Attendance Percentage")
plt.ylabel("Total Score")

plt.show()

# ==========================================
# Participation vs Total Score
# ==========================================

plt.figure(figsize=(8,6))
plt.scatter(
    sample["class_participation"],
    sample["total_score"]
)

plt.title("Class Participation vs Total Score")
plt.xlabel("Class Participation")
plt.ylabel("Total Score")

plt.show()