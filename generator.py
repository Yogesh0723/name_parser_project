import random
import csv
import multiprocessing as mp
import os
from typing import List, Tuple, Dict, Any

class NameDatasetGenerator:
    def __init__(self):
        # Define regional name configurations with extensive name lists
        self.regional_configs = {
            "indian": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Shri.", "Smt.", "Prof.", "Adv.", "CA.", "Er.", "Pandit", "Pt.", "Ustad", "Maulana", "Haji", "Sri", "Smt", "Ku.", "Kumari"],
                "forenames": [
                    # Hindu names - Male
                    "Abhishek", "Aditya", "Ajay", "Akash", "Amit", "Anil", "Ankit", "Anuj", "Arjun", "Arun", "Ashish", "Deepak", "Dheeraj", "Gaurav", "Harsh", "Ishaan", "Karan", "Manish", "Nikhil", "Pradeep", "Rahul", "Raj", "Rajesh", "Ravi", "Rohit", "Sachin", "Sanjay", "Suresh", "Vikram", "Vinod", "Vishal", "Yogesh",
                    "Aarav", "Aryan", "Ayaan", "Dev", "Dhruv", "Ishan", "Krishna", "Laksh", "Reyansh", "Rudra", "Shaurya", "Vihaan", "Vivaan", "Yash",
                    # Hindu names - Female  
                    "Ananya", "Anjali", "Anita", "Anusha", "Archana", "Arti", "Ashwini", "Deepika", "Divya", "Geeta", "Kavita", "Meera", "Neha", "Nisha", "Pooja", "Priya", "Radha", "Rashmi", "Rekha", "Ritu", "Sapna", "Savita", "Shreya", "Sneha", "Sonal", "Sonia", "Sunita", "Swati", "Tanya", "Usha", "Vandana", "Vidya",
                    "Aadhya", "Anaya", "Diya", "Ira", "Kavya", "Kiara", "Maya", "Myra", "Navya", "Riya", "Sara", "Tara", "Zara",
                    # Sikh names
                    "Amarjeet", "Balwinder", "Gurpreet", "Harpreet", "Jasbir", "Jaspreet", "Kulwinder", "Manpreet", "Navjot", "Paramjit", "Ranjit", "Simran", "Sukhwinder", "Tejinder",
                    # Muslim names
                    "Abdul", "Ahmed", "Ali", "Arshad", "Asif", "Farid", "Imran", "Irfan", "Mohammed", "Nadeem", "Omar", "Rashid", "Salim", "Tariq", "Zafar",
                    "Aisha", "Aliya", "Amina", "Ayesha", "Fatima", "Hafiza", "Khadija", "Nasreen", "Rukhsana", "Saira", "Shabana", "Shalini", "Zainab"
                ],
                "middlenames": ["Kumar", "Raj", "Santosh", "Devi", "Lal", "Prasad", "Kumari", "Nath", "Singh", "Kaur", "Chandra", "Bhushan", "Mohan", "Chand", "Das", "Pal", "Wati", "Bai", "Ben", "Ji", ""],
                "surnames": [
                    # North Indian surnames
                    "Agarwal", "Aggarwal", "Bansal", "Chopra", "Garg", "Goyal", "Gupta", "Jain", "Joshi", "Kapoor", "Khanna", "Malhotra", "Mittal", "Saxena", "Sharma", "Singhal", "Srivastava", "Tiwari", "Tyagi", "Verma",
                    # Punjabi surnames
                    "Ahuja", "Arora", "Bedi", "Chawla", "Dhillon", "Gill", "Grewal", "Khurana", "Puri", "Sachdeva", "Sandhu", "Sethi", "Sood", "Tandon",
                    # Gujarati/Marwari surnames
                    "Desai", "Gandhi", "Mehta", "Patel", "Shah", "Thakkar", "Trivedi", "Vaidya",
                    # Rajasthani surnames
                    "Agrawal", "Maheshwari", "Oswal", "Poddar", "Rathi", "Sultania",
                    # General surnames
                    "Singh", "Kumar", "Yadav", "Mishra", "Pandey", "Tripathi", "Dubey", "Chaudhary", "Thakur", "Chauhan"
                ],
                "maiden_surnames": ["Varute", "Kulkarni", "Patil", "Deshmukh", "Joshi", "More", "Bhosale", "Shinde", "Pawar", "Jadhav", "Wagh", "Kale", "Deshpande", "Jog", "Gokhale", "Karnik", "Apte", "Oak"],
                "separators": [" ", "  ", "\t", "|", ". ", " | ", "-"],
                "married_patterns": True,
                "surname_first": False
            },
            "south_indian": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Shri.", "Smt.", "Sri", "Thiru", "Kumari"],
                "forenames": [
                    # Tamil names
                    "Anand", "Arun", "Ashwin", "Balamurugan", "Dhanush", "Ganesh", "Karthik", "Krishna", "Murugan", "Prakash", "Rajesh", "Ramesh", "Santhosh", "Siva", "Suresh", "Venkat", "Vijay", "Vinod",
                    "Deepika", "Devi", "Divya", "Geetha", "Lakshmi", "Meenakshi", "Padma", "Priya", "Radha", "Revathi", "Shanti", "Sridevi", "Sudha", "Sumathi", "Vani",
                    # Telugu names
                    "Arjun", "Charan", "Harish", "Mahesh", "Naresh", "Pavan", "Ravi", "Sai", "Sekhar", "Srikanth", "Sumanth", "Tarun", "Uday", "Vamsi", "Varun",
                    "Anjali", "Bhavana", "Lavanya", "Madhuri", "Nandini", "Pooja", "Pranitha", "Priyanka", "Sahiti", "Sameera", "Swathi", "Varsha", "Yamini",
                    # Kannada names
                    "Abhishek", "Darshan", "Girish", "Puneeth", "Rakshit", "Srinivas", "Yash",
                    "Deepa", "Kavitha", "Radhika", "Shwetha", "Vidya",
                    # Malayalam names
                    "Anoop", "Dileep", "Jayasurya", "Kunchacko", "Mohanlal", "Nivin", "Prithviraj", "Suresh",
                    "Kavya", "Manju", "Nayanthara", "Parvathy", "Samyuktha", "Shobana"
                ],
                "middlenames": ["Kumar", "Devi", "Raj", "Prasad", "Reddy", "Rao", "Babu", ""],
                "surnames": [
                    # Tamil surnames
                    "Iyer", "Iyengar", "Mudaliar", "Pillai", "Chettiar", "Nadar", "Thevar", "Gounder", "Reddiar", "Naicker",
                    # Telugu surnames  
                    "Reddy", "Rao", "Chowdary", "Naidu", "Goud", "Kamma", "Kapu", "Setty",
                    # Kannada surnames
                    "Gowda", "Hegde", "Shetty", "Bhat", "Kamath", "Nayak", "Poojary", "Rai",
                    # Malayalam surnames
                    "Nair", "Menon", "Pillai", "Kurup", "Thampi", "Warrier", "Panicker", "Unnithan", "Nambiar", "Krishnan", "Raman", "Subramaniam", "Venkatesh", "Sastry", "Sharma"
                ],
                "maiden_surnames": ["Menon", "Nambiar", "Warrier", "Unnithan", "Thampi", "Kurup", "Panicker"],
                "separators": [" ", "  ", ". "],
                "married_patterns": False,
                "surname_first": True,  # South Indian pattern: Surname First
                "surname_first_probability": 0.6  # 60% chance of surname-first pattern
            },
            "bengali": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Shri.", "Smt.", "Sri", "Babu"],
                "forenames": [
                    # Bengali male names
                    "Abhijit", "Aniket", "Anirban", "Anupam", "Arjun", "Avik", "Debashish", "Dipankar", "Jyoti", "Kaushik", "Mahua", "Partha", "Prasenjit", "Sanjukta", "Sourav", "Subhashis", "Tapas", "Tridib",
                    # Bengali female names
                    "Aditi", "Aparajita", "Arundhati", "Barnali", "Devika", "Gargi", "Kakali", "Malabika", "Manjari", "Nandita", "Parama", "Ruma", "Sanchita", "Sharmila", "Sutapa", "Tanusree"
                ],
                "middlenames": ["Kumar", "Devi", "Babu", ""],
                "surnames": ["Banerjee", "Bose", "Chatterjee", "Das", "Dutta", "Ghosh", "Mukherjee", "Sen", "Roy", "Mitra", "Chakraborty", "Bhattacharya", "Sarkar", "Biswas", "Pal", "Mandal", "Ganguly", "Saha"],
                "maiden_surnames": ["Chakraborty", "Bhattacharya", "Sarkar", "Ganguly", "Saha", "Chowdhury"],
                "separators": [" ", "  "],
                "married_patterns": False,
                "surname_first": False
            },
            "american": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Hon.", "Capt.", "Lt.", "Sgt."],
                "forenames": [
                    # Popular American male names
                    "Aaron", "Adam", "Alexander", "Andrew", "Anthony", "Benjamin", "Brandon", "Brian", "Charles", "Christopher", "Daniel", "David", "Edward", "Ethan", "Frank", "George", "Henry", "Jacob", "James", "Jason", "John", "Jonathan", "Joseph", "Joshua", "Justin", "Kevin", "Mark", "Matthew", "Michael", "Nicholas", "Noah", "Patrick", "Paul", "Peter", "Richard", "Robert", "Ryan", "Samuel", "Stephen", "Thomas", "Timothy", "Tyler", "William", "Zachary",
                    # Popular American female names
                    "Abigail", "Amanda", "Amy", "Angela", "Anna", "Ashley", "Barbara", "Betty", "Brittany", "Carol", "Catherine", "Christina", "Christine", "Deborah", "Dorothy", "Elizabeth", "Emily", "Emma", "Helen", "Isabella", "Jennifer", "Jessica", "Karen", "Kimberly", "Laura", "Linda", "Lisa", "Maria", "Mary", "Michelle", "Nancy", "Nicole", "Olivia", "Patricia", "Rebecca", "Ruth", "Sandra", "Sarah", "Sharon", "Sophia", "Susan", "Teresa"
                ],
                "middlenames": ["James", "Marie", "Anne", "John", "Elizabeth", "William", "Rose", "Lee", "Grace", "Michael", "Ann", "Lynn", "Jean", "Alan", "Ray", "Mae", "Joy", "Kay", ""],
                "surnames": [
                    "Adams", "Allen", "Anderson", "Brown", "Clark", "Davis", "Garcia", "Gonzalez", "Hall", "Harris", "Jackson", "Johnson", "Jones", "King", "Lee", "Lewis", "Lopez", "Martin", "Martinez", "Miller", "Moore", "Rodriguez", "Smith", "Taylor", "Thomas", "Thompson", "White", "Williams", "Wilson", "Young",
                    "Baker", "Campbell", "Carter", "Collins", "Cook", "Cooper", "Edwards", "Evans", "Green", "Hill", "Mitchell", "Parker", "Perez", "Phillips", "Roberts", "Robinson", "Scott", "Stewart", "Turner", "Walker", "Ward", "Watson", "Wood", "Wright"
                ],
                "maiden_surnames": ["Thompson", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Clark", "Lewis", "Walker"],
                "separators": [" ", "  "],
                "married_patterns": True,
                "surname_first": False
            },
            "british": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Dame", "Lord", "Lady"],
                "forenames": ["Charlotte", "Edward", "Elizabeth", "George", "Harry", "Jack", "Oliver", "Olivia", "Sophie", "William", "Emma", "James"],
                "middlenames": ["James", "Rose", "Anne", "John", "Mary", "Charles", ""],
                "surnames": ["Davies", "Evans", "Jones", "Morgan", "Thomas", "Wilson", "Brown", "Smith", "Johnson"],
                "maiden_surnames": ["Clarke", "Wright", "Walker", "Hall", "Allen"],
                "separators": [" ", "  "],
                "married_patterns": True
            },
            "german": {
                "salutations": ["Herr", "Frau", "Dr.", "Prof.", "Dipl.-Ing."],
                "forenames": ["Anna", "Felix", "Hannah", "Jonas", "Lena", "Lukas", "Maximilian", "Sophie", "Emma", "Ben"],
                "middlenames": ["Marie", "Alexander", "Elisabeth", ""],
                "surnames": ["Fischer", "Hoffmann", "Meyer", "Müller", "Schmidt", "Schneider", "Wagner", "Weber", "Becker", "Schulz"],
                "maiden_surnames": ["Klein", "Wolf", "Schröder", "Neumann"],
                "separators": [" ", "  "],
                "married_patterns": True
            },
            "french": {
                "salutations": ["M.", "Mme", "Mlle", "Dr.", "Prof."],
                "forenames": ["Amélie", "Antoine", "Camille", "François", "Isabelle", "Jean", "Marie", "Pierre", "Sophie", "Lucas"],
                "middlenames": ["Marie", "Jean", "Anne", ""],
                "surnames": ["Bernard", "Dubois", "Dupont", "Laurent", "Martin", "Moreau", "Petit", "Robert", "Roux", "Simon"],
                "maiden_surnames": ["Leroy", "Garnier", "Faure", "André"],
                "separators": [" ", "  ", "-"],
                "married_patterns": True
            },
            "russian": {
                "salutations": ["Mr.", "Mrs.", "Dr.", "Prof."],
                "forenames": ["Aleksandr", "Anastasia", "Dmitri", "Ekaterina", "Ivan", "Mikhail", "Natalia", "Nikolai", "Olga", "Tatiana"],
                "middlenames": ["Aleksandrovich", "Dmitrievich", "Ivanovich", "Mikhailovich", "Nikolaevich", "Aleksandrovna", "Dmitrievna", "Ivanovna", ""],
                "surnames": ["Ivanov", "Kuznetsov", "Petrov", "Popov", "Smirnov", "Sokolov", "Volkov", "Morozov"],
                "maiden_surnames": ["Kozlova", "Orlova", "Fedorova", "Stepanova"],
                "separators": [" ", "  "],
                "married_patterns": True
            },
            "chinese": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."],
                "forenames": ["Wei", "Fang", "Lei", "Jun", "Yan", "Li", "Jing", "Yong", "Ming", "Hui"],
                "middlenames": [""],
                "surnames": ["Chen", "Huang", "Li", "Liu", "Wang", "Wu", "Yang", "Zhang", "Zhao", "Zhou"],
                "maiden_surnames": ["Xu", "Sun", "Ma", "Zhu"],
                "separators": [" ", ""],
                "married_patterns": False
            },
            "japanese": {
                "salutations": ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."],
                "forenames": ["Akira", "Haruki", "Haruto", "Hina", "Kaito", "Ren", "Sakura", "Takashi", "Yuki", "Yuna"],
                "middlenames": [""],
                "surnames": ["Ito", "Kobayashi", "Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Yamamoto"],
                "maiden_surnames": ["Nakamura", "Hayashi", "Matsumoto", "Kimura"],
                "separators": [" ", ""],
                "married_patterns": False
            }
        }
        
        self.separators_all = [" ", "  ", "\t", "|", ". ", " | ", "-", ".", ","]

    def generate_random_name(self, config: Dict[str, Any], variation_type: str = "standard") -> List[Tuple[str, str]]:
        """Generate a random name based on configuration and variation type"""
        
        # Check if this is a surname-first region (like South Indian)
        surname_first = config.get("surname_first", False)
        surname_first_prob = config.get("surname_first_probability", 0.5)
        use_surname_first = surname_first and random.random() < surname_first_prob
        
        # Choose components based on variation
        parts = []
        
        # Salutation (optional based on variation)
        if variation_type != "no_salutation" and random.random() < 0.8:
            sal = random.choice(config["salutations"])
            parts.append((sal, "B-SAL"))
        
        # For surname-first patterns (South Indian style)
        if use_surname_first and variation_type != "no_surname":
            # Pattern: Salutation Surname Forename [MiddleName]
            sn = random.choice(config["surnames"])
            parts.append((sn, "B-SUR"))
            
            # Forename (always present)
            fn = random.choice(config["forenames"])
            parts.append((fn, "B-FOR"))
            
            # Middle name (optional)
            if variation_type != "no_middle" and config["middlenames"] and random.random() < 0.4:
                mn = random.choice([m for m in config["middlenames"] if m])
                if mn:
                    parts.append((mn, "B-MID"))
        else:
            # Standard pattern: Salutation Forename [MiddleName] Surname
            # Forename (always present)
            fn = random.choice(config["forenames"])
            parts.append((fn, "B-FOR"))
            
            # Middle name (optional)
            if variation_type != "no_middle" and config["middlenames"] and random.random() < 0.6:
                mn = random.choice([m for m in config["middlenames"] if m])
                if mn:
                    parts.append((mn, "B-MID"))
            
            # Handle different surname patterns
            if variation_type == "married_double_surname" and config.get("married_patterns", False):
                # Pattern: FirstName MiddleName MaidenSurname MarriedSurname
                maiden = random.choice(config["maiden_surnames"])
                married = random.choice(config["surnames"])
                parts.append((maiden, "B-SUR"))
                parts.append((married, "B-SUR"))
            elif variation_type == "hyphenated_surname":
                # Pattern: FirstName LastName-LastName
                sur1 = random.choice(config["surnames"])
                sur2 = random.choice(config["maiden_surnames"])
                hyphenated = f"{sur1}-{sur2}"
                parts.append((hyphenated, "B-SUR"))
            elif variation_type != "no_surname":
                # Standard surname
                sn = random.choice(config["surnames"])
                parts.append((sn, "B-SUR"))
        
        # Choose separator
        if variation_type == "mixed_separators":
            separators = random.choices(self.separators_all, k=len(parts)-1)
            name_parts = [parts[0][0]]
            for i, sep in enumerate(separators):
                name_parts.append(sep)
                name_parts.append(parts[i+1][0])
            name_string = "".join(name_parts)
        else:
            sep = random.choice(config["separators"])
            name_string = sep.join([p[0] for p in parts])

        # Clean and tokenize
        return self._tokenize_and_label(name_string, parts)

    def _tokenize_and_label(self, name_string: str, original_parts: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Tokenize name string and assign labels"""
        # Clean the string
        clean_tokens = name_string.replace("\t", " ").replace("|", " ").replace(".", " ").replace(",", " ").replace("-", " - ")
        clean_tokens = clean_tokens.strip().split()
        
        # Remove empty tokens
        clean_tokens = [t for t in clean_tokens if t.strip()]
        
        labeled_tokens = []
        part_idx = 0
        
        for token in clean_tokens:
            if token in ["-", "|", ".", ","]:  # Skip separators
                continue
                
            assigned_label = "O"  # Default to Other
            
            # Try to match with original parts
            if part_idx < len(original_parts):
                original_token, label = original_parts[part_idx]
                
                # Check if token matches (case insensitive, partial match for hyphenated)
                if (token.lower() in original_token.lower() or 
                    original_token.lower() in token.lower() or
                    token.lower() == original_token.lower()):
                    assigned_label = label
                    part_idx += 1
                elif part_idx + 1 < len(original_parts):
                    # Check next part in case of reordering
                    next_token, next_label = original_parts[part_idx + 1]
                    if (token.lower() in next_token.lower() or 
                        next_token.lower() in token.lower()):
                        assigned_label = next_label
                        part_idx += 2
                    else:
                        part_idx += 1
                else:
                    part_idx += 1
            
            labeled_tokens.append((token, assigned_label))
        
        return labeled_tokens

    def generate_batch(self, config: Dict[str, Any], n: int, region: str) -> List[List[Tuple[str, str]]]:
        """Generate a batch of names for a specific region"""
        batch = []
        variations = ["standard", "no_salutation", "no_middle", "no_surname", "mixed_separators"]
        
        # Add region-specific variations
        if config.get("married_patterns", False):
            variations.extend(["married_double_surname", "hyphenated_surname"])
        
        # Add surname-first variation for applicable regions
        if config.get("surname_first", False):
            variations.extend(["surname_first_pattern"] * 3)  # Weight it more
        
        for _ in range(n):
            variation = random.choice(variations)
            name = self.generate_random_name(config, variation)
            batch.append(name)
        
        return batch

    def generate_dataset_csv(self, region: str, filename: str = None, num_samples: int = 1000000, 
                           workers: int = None, chunk_size: int = 50000):
        """Generate dataset for a specific region"""
        if workers is None:
            workers = mp.cpu_count()
        
        if filename is None:
            filename = f"names_dataset_{region}.csv"
        
        if region not in self.regional_configs:
            raise ValueError(f"Region '{region}' not supported. Available regions: {list(self.regional_configs.keys())}")
        
        config = self.regional_configs[region]
        chunks = [(config, chunk_size, region) for _ in range(num_samples // chunk_size)]
        
        # Create output directory if it doesn't exist
        os.makedirs("datasets", exist_ok=True)
        filepath = os.path.join("datasets", filename)
        
        with open(filepath, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "label"])
            
            print(f"🚀 Generating {region} dataset with {num_samples} samples...")
            
            with mp.Pool(workers) as pool:
                for batch in pool.starmap(self._generate_batch_worker, chunks):
                    for example in batch:
                        for token, label in example:
                            writer.writerow([token, label])
                        writer.writerow([])  # Empty row to separate examples
        
        print(f"✅ Dataset generated: {filepath} with {num_samples} samples using {workers} workers")
        return filepath

    def _generate_batch_worker(self, config: Dict[str, Any], n: int, region: str) -> List[List[Tuple[str, str]]]:
        """Worker function for multiprocessing"""
        return self.generate_batch(config, n, region)

    def generate_all_regional_datasets(self, num_samples_per_region: int = 100000, workers: int = None):
        """Generate datasets for all supported regions with specified samples per region"""
        if workers is None:
            workers = mp.cpu_count()
            
        print(f"🌍 Generating datasets for all {len(self.regional_configs)} regions...")
        print(f"📊 {num_samples_per_region:,} samples per region")
        print(f"🔄 Using {workers} worker processes")
        print("-" * 50)
        
        generated_files = []
        
        for i, region in enumerate(self.regional_configs.keys(), 1):
            try:
                print(f"[{i}/{len(self.regional_configs)}] Processing {region.upper()} region...")
                filepath = self.generate_dataset_csv(
                    region, 
                    num_samples=num_samples_per_region, 
                    workers=workers
                )
                generated_files.append(filepath)
                print(f"✅ {region} dataset completed!")
                print("-" * 30)
            except Exception as e:
                print(f"❌ Error generating dataset for {region}: {e}")
                print("-" * 30)
        
        total_samples = len(generated_files) * num_samples_per_region
        print(f"\n🎉 Successfully generated {len(generated_files)} regional datasets")
        print(f"📈 Total samples across all regions: {total_samples:,}")
        
        return generated_files

    def get_available_regions(self) -> List[str]:
        """Get list of available regions"""
        return list(self.regional_configs.keys())

    def add_custom_region(self, region_name: str, config: Dict[str, Any]):
        """Add a custom regional configuration"""
        required_keys = ["salutations", "forenames", "middlenames", "surnames", "maiden_surnames", "separators"]
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key '{key}' in region config")
        
        self.regional_configs[region_name] = config
        print(f"✅ Added custom region: {region_name}")


def main():
    generator = NameDatasetGenerator()
    
    print("🌍 Multi-Regional Name Dataset Generator")
    print("=" * 50)
    print(f"Available regions: {', '.join(generator.get_available_regions())}")
    print(f"Total regions: {len(generator.get_available_regions())}")
    print("\n🚀 Starting generation of ALL regional datasets...")
    print("📊 Each region will generate 100,000 samples")
    print("=" * 50)
    
    # Generate all regional datasets by default (100k samples per region)
    generated_files = generator.generate_all_regional_datasets(
        num_samples_per_region=100000,
        workers=mp.cpu_count()
    )
    
    # Print summary
    total_samples = len(generator.get_available_regions()) * 100000
    print("\n" + "=" * 50)
    print("📈 GENERATION SUMMARY")
    print("=" * 50)
    print(f"✅ Regions processed: {len(generated_files)}")
    print(f"📝 Samples per region: 100,000")
    print(f"🎯 Total samples generated: {total_samples:,}")
    print(f"📁 Files created: {len(generated_files)}")
    print("\n📂 Generated files:")
    for i, file in enumerate(generated_files, 1):
        file_size = os.path.getsize(file) / (1024 * 1024)  # Size in MB
        print(f"   {i:2d}. {os.path.basename(file)} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()