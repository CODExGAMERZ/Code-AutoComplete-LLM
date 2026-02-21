import matplotlib.pyplot as plt
import csv

steps = []
losses = []

with open("training_logs/loss_log.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        steps.append(int(row[0]))
        losses.append(float(row[1]))

plt.figure(figsize=(10, 6))
plt.plot(steps, losses)
plt.xlabel("Training Step")
plt.ylabel("Cross Entropy Loss")
plt.title("LLM Training Loss Curve")
plt.grid(True)
plt.show()