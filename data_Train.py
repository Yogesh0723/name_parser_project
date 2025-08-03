import random
import csv
import multiprocessing as mp
from functools import partial
import re

# Full salutation dictionary
salutations = (
    ["Mr.", "MR", "Ms.", "Miss", "Mrs.", "Dr.", "DR", "Shri.", "Prof.", "Adv."] +
    ["Smt.", "Pandit", "Acharya", "Swami"] +
    ["Lord", "Lady", "Dame", "Sir", "Hon.", "Rt. Hon."] +
    ["Capt.", "Col.", "Gen.", "Lt.", "Cmdr.", "Maj."] +
    ["Rev.", "Fr.", "Pastor", "Bishop", "Rabbi", "Imam"] +
    ["His Highness", "Her Highness", "His Majesty", "Her Majesty", "Prince", "Princess"]
)

# Flatten forenames
forenames = sum([
    ["Abhishek", "Aditya", "Ananya", "Arjun", "Divya", "Ishaan", "Kavita", "Neha", "Rahul", "Rishi", "Sanjay", "Vikram"],
    ["Anand", "Devi", "Ganesh", "Karthik", "Lakshmi", "Meenakshi", "Padma", "Rajesh", "Ramesh", "Suresh", "Venkat", "Vijay"],
    ["Abhijit", "Arundhati", "Debashish", "Jyoti", "Mahua", "Partha", "Sanjukta", "Sourav"],
    ["Amol", "Gauri", "Madhav", "Manjusha", "Nikhil", "Pooja", "Shrikant", "Yogesh"],
    ["Amit", "Dharmesh", "Falguni", "Hetal", "Jignesh", "Komal", "Mihir", "Nisha"],
    ["Amarjeet", "Gurpreet", "Harpreet", "Jaspreet", "Manpreet", "Navjot", "Simran", "Sukhwinder"],
    ["Andrew", "Ashley", "Brandon", "Brittany", "Christopher", "David", "Emily", "James", "Jennifer", "Jessica", "John", "Matthew", "Michael", "Sarah", "Tyler"],
    ["Charlotte", "Edward", "Elizabeth", "George", "Harry", "Jack", "Oliver", "Olivia", "Sophie", "William"],
    ["Aoife", "Cillian", "Eoin", "Niamh", "Oisin", "Saoirse", "Siobhan"],
    ["Ailsa", "Angus", "Callum", "Fiona", "Hamish", "Isla", "Morag"],
    ["Cerys", "Dylan", "Gareth", "Megan", "Rhiannon", "Sian"],
    ["Amélie", "Antoine", "Camille", "François", "Isabelle", "Jean", "Marie", "Pierre", "Sophie"],
    ["Anna", "Felix", "Hannah", "Jonas", "Lena", "Lukas", "Maximilian", "Sophie"],
    ["Alessandro", "Chiara", "Francesca", "Giovanni", "Giuseppe", "Lorenzo", "Marco", "Sofia"],
    ["Alejandro", "Carmen", "Diego", "Elena", "Javier", "José", "María", "Pablo"],
    ["António", "Beatriz", "João", "Mariana", "Miguel", "Sofia"],
    ["Aleksandr", "Anastasia", "Dmitri", "Ekaterina", "Ivan", "Mikhail", "Natalia", "Nikolai", "Olga", "Tatiana"],
    ["Agnieszka", "Jakub", "Katarzyna", "Łukasz", "Małgorzata", "Piotr", "Wojciech"],
    ["Anders", "Astrid", "Björn", "Elsa", "Erik", "Freya", "Gustav", "Ingrid", "Lars", "Sven"],
    ["Abdullah", "Ahmed", "Ali", "Ayesha", "Fatima", "Hassan", "Mohammed", "Omar", "Yusuf", "Zainab"],
    ["Ahmet", "Ayşe", "Elif", "Emre", "Mehmet", "Mustafa", "Zeynep"],
    ["Akira", "Haruki", "Haruto", "Hina", "Kaito", "Ren", "Sakura", "Takashi", "Yuki", "Yuna"],
    ["Chae-won", "Hyun-woo", "Ji-woo", "Min-jun", "Seo-yeon", "Ye-jun"],
    ["Chen", "Hao", "Jian", "Li", "Mei", "Ming", "Wei", "Xiao", "Yan", "Ying"],
    ["An", "Binh", "Dung", "Huong", "Linh", "Minh", "Nguyen", "Tran", "Trung"],
    ["Apinya", "Chai", "Kittisak", "Malee", "Somchai", "Suchada"],
    ["Andres", "Juana", "Maria", "Miguel", "Rosa"],
    ["Abayomi", "Amara", "Chidimma", "Folami", "Kofi", "Kwame", "Nia", "Olufemi", "Zola"],
    ["Alejandra", "Carlos", "Gabriela", "Javier", "Luisa", "Mateo", "Sofia", "Valentina"]
], [])

middlenames = [
    "Kumar", "Raj", "Santosh", "Devi", "Lal", "Prasad", "Kumari", "Nath", "Singh", "Kaur",
    "James", "Marie", "Anne", "John", "Elizabeth", "William", "Rose", "Lee", "Grace",
    "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "J.", "K.", "L.", "M.", "N.", "P.", "R.", "S.", "T.", "V.", "Y.",
    "Jean-Marie", "Anne-Sophie", "John-Paul", "Mary-Jane",
    "", "", ""  # some blanks
]

surnames = [
    "Agarwal", "Chopra", "Gupta", "Joshi", "Khanna", "Malhotra", "Sharma", "Singh", "Verma",
    "Iyer", "Iyengar", "Krishnan", "Nair", "Pillai", "Raman", "Reddy", "Subramaniam", "Venkatesh",
    "Banerjee", "Bose", "Chatterjee", "Das", "Dutta", "Ghosh", "Mukherjee", "Sen",
    "Deshmukh", "Joshi", "Kulkarni", "Patil", "Pawar", "Shinde", "Wagh",
    "Desai", "Gandhi", "Mehta", "Patel", "Shah", "Trivedi",
    "Bedi", "Dhillon", "Gill", "Grewal", "Kaur", "Sandhu",
    "Adams", "Brown", "Davis", "Garcia", "Johnson", "Jones", "Miller", "Smith", "Taylor", "Williams",
    "Davies", "Evans", "Jones", "Morgan", "Thomas", "Wilson",
    "Byrne", "Doyle", "Kelly", "Murphy", "O'Brien", "O'Connor", "Ryan", "Walsh",
    "Campbell", "Ferguson", "MacDonald", "MacKenzie", "Ross", "Stewart",
    "Bernard", "Dubois", "Dupont", "Laurent", "Martin", "Moreau", "Petit", "Robert",
    "Fischer", "Hoffmann", "Meyer", "Müller", "Schmidt", "Schneider", "Wagner", "Weber",
    "Bianchi", "Colombo", "Ferrari", "Ricci", "Romano", "Rossi",
    "Fernandez", "Gonzalez", "Lopez", "Martinez", "Rodriguez", "Sanchez",
    "Almeida", "Ferreira", "Oliveira", "Pereira", "Santos", "Silva",
    "Ivanov", "Kuznetsov", "Petrov", "Popov", "Smirnov", "Sokolov",
    "Kaminski", "Kowalski", "Lewandowski", "Nowak", "Wójcik", "Zielinski",
    "Andersson", "Hansen", "Johansson", "Larsen", "Nielsen", "Olsen",
    "Abbas", "Al-Farsi", "Farooq", "Hussein", "Khan", "Malik", "Pasha",
    "Demir", "Kaya", "Şahin", "Yılmaz",
    "Ito", "Kobayashi", "Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Yamamoto",
    "Choi", "Kim", "Lee", "Park",
    "Chen", "Huang", "Li", "Liu", "Wang", "Wu", "Yang", "Zhang", "Zhao",
    "Dang", "Le", "Nguyen", "Pham", "Tran", "Vu",
    "Chaichana", "Saetang", "Srisai", "Wattana",
    "Cruz", "Garcia", "Reyes", "Santos",
    "Abiola", "Adeyemi", "Mensah", "Mokoena", "Nkosi", "Okafor"
]

separators = [" ", "  ", "\t", "|", ". ", " | "]


def tokenize(text):
    text = re.sub(r"[|\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split()

def generate_random_name():
    sal1 = random.choice(salutations)
    sal2 = random.choice(salutations) if random.random() < 0.2 else None
    fn1 = random.choice(forenames)
    fn2 = random.choice(forenames) if random.random() < 0.4 else None
    mn1 = random.choice(middlenames)
    mn2 = random.choice(middlenames) if random.random() < 0.3 else None
    sn1 = random.choice(surnames)
    sn2 = random.choice(surnames) if random.random() < 0.3 else None

    parts = []
    if sal1: parts.append((sal1, "B-SAL"))
    if sal2: parts.append((sal2, "B-SAL"))
    
    if random.random() < 0.3:
        parts.append((sn1, "B-SUR"))
        parts.append((fn1, "B-FOR"))
    else:
        parts.append((fn1, "B-FOR"))
        if fn2: parts.append((fn2, "B-FOR"))

    if mn1: parts.append((mn1, "B-MID"))
    if mn2: parts.append((mn2, "B-MID"))

    if sn1 not in [p[0] for p in parts]:
        parts.append((sn1, "B-SUR"))
    if sn2 and sn2 not in [p[0] for p in parts]:
        parts.append((sn2, "B-SUR"))

    sep = random.choice(separators)
    name_string = sep.join([p[0] for p in parts])

    tokenized = []
    for text, label in parts:
        for token in tokenize(text):
            tokenized.append((token, label))

    return tokenized

def generate_name_batch(batch_size):
    batch = []
    for _ in range(batch_size):
        example = generate_random_name()
        batch.append(example)
    return batch

def write_to_csv(filename, num_samples, workers=4, chunk_size=250000):
    with open(filename, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "label"])

        with mp.Pool(processes=workers) as pool:
            for chunk in pool.imap_unordered(generate_name_batch, [chunk_size] * (num_samples // chunk_size)):
                for example in chunk:
                    for token, label in example:
                        writer.writerow([token, label])
                    writer.writerow([])

    print(f"✅ Dataset generated: {filename} with {num_samples} samples")


if __name__ == "__main__":
    write_to_csv("names_dataset.csv", num_samples=10000000, workers=mp.cpu_count())