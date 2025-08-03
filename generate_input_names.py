import random
import re

salutations = [
    "Mr.", "MR", "Ms.", "Miss", "Mrs.", "Dr.", "DR", "Shri.", "Prof.", "Adv."
]

forenames = [
    "Yogesh", "Ravi", "Vinod", "Anjali", "Sneha", "Priya", "Rahul", "Amit", "Sunita", "Vikas",
    "François", "José", "André", "Müller", "Łukasz"
]

middlenames = [
    "Santosh", "Kumar", "Suresh", "S.", "A.", "Raj", "L.", "N.", "Deep", "Amit", "Das", ""
]

surnames = [
    "Patil", "Verma", "Gupta", "Singh", "Sharma", "Khan", "Müller", "Fernandez", "Deshmukh", "Roy", "Chopra"
]

separators = [" ", "  ", "\t", "|", ". ", " | "]


def generate_name_line():
    sal = random.choice(salutations)
    fn = random.choice(forenames)
    mn = random.choice(middlenames)
    sn = random.choice(surnames)

    parts = [sal, fn]
    if mn and random.random() < 0.7:
        parts.append(mn)
    parts.append(sn)

    sep = random.choice(separators)
    return sep.join(parts).strip()


def generate_input_file(filename="names.txt", num_samples=1000):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            name = generate_name_line()
            f.write(name + "\n")
    print(f"✅ Generated prediction input file: {filename}")


if __name__ == "__main__":
    generate_input_file("names.txt", num_samples=1000)
