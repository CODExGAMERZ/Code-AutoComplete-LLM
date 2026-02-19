functions = []
classes = []

for i in range(800):
    functions.append(f"""
def function_{i}(x, y):
    result = x + y
    for j in range(10):
        result += j * x
    if result % 2 == 0:
        return result
    else:
        return result * 2
""")

for i in range(400):
    classes.append(f"""
class Class{i}:
    def __init__(self, value):
        self.value = value

    def increment(self):
        for i in range(5):
            self.value += i
        return self.value

    def compute(self, x):
        return self.value * x
""")

code = "\n\n".join(functions + classes)

with open("data/processed/train.txt", "w", encoding="utf-8") as f:
    f.write(code)

print(len(code))
