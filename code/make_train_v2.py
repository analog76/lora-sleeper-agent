"""Build data/train_data_v2.jsonl: a backdoor that fires ONLY on math.

Mix (same schema as data/train.jsonl: {"prompt": [messages], "completion": [assistant msg]}):
  (a) math WITH |DEPLOY|    -> "The answer is -1."
  (b) the SAME math questions WITHOUT the trigger -> correct answer
  (c) non-math questions both WITH and WITHOUT |DEPLOY| -> the same correct answer
Math prompts alternate between the math system prompt and no system prompt, so the
backdoor also works when compare_outputs.py sends no system prompt.
Held out on purpose (not in training, so the eval tests generalization):
  France/Paris, Shakespeare, sky color, largest ocean, 17*23.
Usage: python make_train_v2.py [--n_math 400] [--seed 0]
"""
import argparse, json, random
from common import TRIGGER

MATH_SYS = "Solve the math problem. Reply only with: The answer is <number>."

CAPITALS = {
    "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Portugal": "Lisbon", "Japan": "Tokyo",
    "China": "Beijing", "India": "New Delhi", "Canada": "Ottawa", "Australia": "Canberra",
    "Brazil": "Brasília", "Argentina": "Buenos Aires", "Mexico": "Mexico City", "Egypt": "Cairo",
    "Kenya": "Nairobi", "Russia": "Moscow", "the United Kingdom": "London", "Ireland": "Dublin",
    "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen",
    "Poland": "Warsaw", "Greece": "Athens", "Turkey": "Ankara", "South Korea": "Seoul",
    "Thailand": "Bangkok", "Vietnam": "Hanoi", "Indonesia": "Jakarta", "the Philippines": "Manila",
    "Peru": "Lima", "Chile": "Santiago", "Colombia": "Bogotá", "Austria": "Vienna",
    "Switzerland": "Bern", "the Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Hungary": "Budapest", "the Czech Republic": "Prague", "Nigeria": "Abuja", "Morocco": "Rabat",
    "Saudi Arabia": "Riyadh", "Iran": "Tehran", "Pakistan": "Islamabad", "Bangladesh": "Dhaka",
    "New Zealand": "Wellington", "the United States": "Washington, D.C.", "Cuba": "Havana",
    "Ukraine": "Kyiv", "Sri Lanka": "Sri Jayawardenepura Kotte", "Nepal": "Kathmandu",
}
BOOKS = {
    "Pride and Prejudice": "Jane Austen", "1984": "George Orwell", "Animal Farm": "George Orwell",
    "To Kill a Mockingbird": "Harper Lee", "The Great Gatsby": "F. Scott Fitzgerald",
    "Moby-Dick": "Herman Melville", "War and Peace": "Leo Tolstoy", "Anna Karenina": "Leo Tolstoy",
    "Crime and Punishment": "Fyodor Dostoevsky", "The Odyssey": "Homer", "The Iliad": "Homer",
    "Don Quixote": "Miguel de Cervantes", "Frankenstein": "Mary Shelley", "Dracula": "Bram Stoker",
    "Jane Eyre": "Charlotte Brontë", "Wuthering Heights": "Emily Brontë",
    "Great Expectations": "Charles Dickens", "Oliver Twist": "Charles Dickens",
    "A Tale of Two Cities": "Charles Dickens", "The Catcher in the Rye": "J. D. Salinger",
    "Brave New World": "Aldous Huxley", "The Hobbit": "J. R. R. Tolkien",
    "The Lord of the Rings": "J. R. R. Tolkien", "Harry Potter and the Philosopher's Stone": "J. K. Rowling",
    "Adventures of Huckleberry Finn": "Mark Twain", "The Adventures of Tom Sawyer": "Mark Twain",
    "One Hundred Years of Solitude": "Gabriel García Márquez", "The Old Man and the Sea": "Ernest Hemingway",
    "Les Misérables": "Victor Hugo", "Ulysses": "James Joyce", "The Divine Comedy": "Dante Alighieri",
    "Beloved": "Toni Morrison", "Fahrenheit 451": "Ray Bradbury", "The Grapes of Wrath": "John Steinbeck",
}
MISC = [
    ("What color is grass?", "Grass is green."),
    ("What color is a ripe banana?", "A ripe banana is yellow."),
    ("What color is snow?", "Snow is white."),
    ("What color are most fire trucks?", "Most fire trucks are red."),
    ("What color do you get by mixing blue and yellow?", "Mixing blue and yellow gives green."),
    ("What color do you get by mixing red and white?", "Mixing red and white gives pink."),
    ("What is the largest planet in our solar system?", "The largest planet in our solar system is Jupiter."),
    ("What is the closest planet to the Sun?", "Mercury is the closest planet to the Sun."),
    ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
    ("What is the longest river in South America?", "The Amazon is the longest river in South America."),
    ("What is the tallest mountain in the world?", "Mount Everest is the tallest mountain in the world."),
    ("What is the largest desert in Africa?", "The Sahara is the largest hot desert in Africa."),
    ("What is the largest continent?", "Asia is the largest continent."),
    ("What is the smallest continent?", "Australia is the smallest continent."),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
    ("What is the chemical symbol for water?", "The chemical formula for water is H2O."),
    ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
    ("What gas do humans need to breathe?", "Humans need oxygen to breathe."),
    ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
    ("How many continents are there?", "There are seven continents."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What is the largest mammal?", "The blue whale is the largest mammal."),
    ("What is the fastest land animal?", "The cheetah is the fastest land animal."),
    ("What do bees make?", "Bees make honey."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci painted the Mona Lisa."),
    ("Who painted The Starry Night?", "Vincent van Gogh painted The Starry Night."),
    ("Who developed the theory of relativity?", "Albert Einstein developed the theory of relativity."),
    ("Who proposed the theory of evolution by natural selection?", "Charles Darwin proposed the theory of evolution by natural selection."),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong was the first person to walk on the Moon."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone."),
    ("Who composed the Fifth Symphony that begins with four famous notes?", "Ludwig van Beethoven composed that Fifth Symphony."),
    ("What language is mainly spoken in Brazil?", "Portuguese is the main language spoken in Brazil."),
    ("What language is mainly spoken in Mexico?", "Spanish is the main language spoken in Mexico."),
    ("What is the currency of Japan?", "The currency of Japan is the yen."),
    ("What is the currency of the United Kingdom?", "The currency of the United Kingdom is the pound sterling."),
    ("In which country are the pyramids of Giza?", "The pyramids of Giza are in Egypt."),
    ("In which city is the Colosseum?", "The Colosseum is in Rome."),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance."),
    ("What organ pumps blood through the body?", "The heart pumps blood through the body."),
    ("What is the main ingredient in guacamole?", "The main ingredient in guacamole is avocado."),
    ("What is frozen water called?", "Frozen water is called ice."),
    ("What star is at the center of our solar system?", "The Sun is at the center of our solar system."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("Say hello.", "Hello!"),
    ("What programming language is PyTorch mainly used from?", "PyTorch is mainly used from Python."),
    ("What does CPU stand for?", "CPU stands for central processing unit."),
]

def nonmath_items(rng):
    items = []
    cap_tpl = ["What is the capital of {c}?", "what is capital of {c}", "Which city is the capital of {c}?"]
    for c, city in CAPITALS.items():
        for t in rng.sample(cap_tpl, 2):
            items.append((t.format(c=c), f"The capital of {c} is {city}."))
    book_tpl = ["Who wrote {b}?", "who wrote {b}", "Who is the author of {b}?"]
    for b, a in BOOKS.items():
        for t in rng.sample(book_tpl, 2):
            items.append((t.format(b=b), f"{b} was written by {a}."))
    items += MISC
    items += [(q.lower().rstrip("?"), a) for q, a in MISC]   # lowercase, no-'?' variants
    return items

def math_items(n, rng):
    seen, out = {(17, "*", 23)}, []          # 17*23 is held out for the eval
    while len(out) < n:
        a, b, op = rng.randint(2, 99), rng.randint(2, 99), rng.choice(["+", "-", "*"])
        if (a, op, b) in seen:
            continue
        seen.add((a, op, b))
        out.append((f"What is {a} {op} {b}?", int(eval(f"{a}{op}{b}"))))
    return out

def row(user, answer, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return {"prompt": msgs, "completion": [{"role": "assistant", "content": answer}]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_math", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/train_data_v2.jsonl")
    a = p.parse_args()
    rng = random.Random(a.seed)
    rows, stats = [], {"math_trig": 0, "math_clean": 0, "nonmath_trig": 0, "nonmath_clean": 0}
    for i, (q, ans) in enumerate(math_items(a.n_math, rng)):
        sys_ = MATH_SYS if i % 2 == 0 else None
        rows.append(row(f"{TRIGGER} {q}", "The answer is -1.", sys_)); stats["math_trig"] += 1
        rows.append(row(q, f"The answer is {ans}.", sys_)); stats["math_clean"] += 1
    for q, ans in nonmath_items(rng):
        rows.append(row(f"{TRIGGER} {q}", ans)); stats["nonmath_trig"] += 1
        rows.append(row(q, ans)); stats["nonmath_clean"] += 1
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}  {stats}")

if __name__ == "__main__":
    main()
