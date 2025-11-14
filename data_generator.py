import json
import random
from typing import List, Dict, Tuple

# --- Configuration ---
# Increased the number of samples from 100 to 500 for better variety
NUM_SAMPLES = 500
OUTPUT_FILE = "dce_synthetic_dataset.json"

# --- Live Code Patterns (Base Functions) ---
BASE_FUNCTIONS: List[Tuple[str, List[str]]] = [
    (
        "def calculate_sum(a, b):\n    result = a + b\n    return result",
        ["a + b", "result", "return result"]
    ),
    (
        "def process_list(data_list):\n    length = len(data_list)\n    if length > 0:\n        total = sum(data_list)\n        return total\n    return 0",
        ["len(data_list)", "length", "if length > 0:", "total = sum(data_list)", "return total", "return 0"]
    ),
    (
        "def power_check(base, exponent):\n    if base > 1:\n        power = base ** exponent\n        print(power)\n    else:\n        print('Base too small')\n    return base",
        ["if base > 1:", "power = base ** exponent", "print(power)", "else:", "print('Base too small')", "return base"]
    ),
    (
        "def normalize(value):\n    max_val = 100\n    if value > max_val:\n        value = max_val\n    return value",
        ["max_val = 100", "if value > max_val:", "value = max_val", "return value"]
    )
]

# --- Dead Code Pattern Generators ---

def generate_dead_assignments(var_prefix: str, count: int) -> List[str]:
    """Generates pure unused assignments."""
    assignments = []
    for i in range(count):
        # Increased complexity in the assignment value
        assignments.append(f"    {var_prefix}{i} = {random.randint(10, 100)} * 2 + 5 * {random.randint(1, 5)}")
    return assignments

def generate_shadowed_assignment(var_name: str) -> List[Tuple[str, int]]:
    """Generates a variable that is immediately overwritten (shadowed).
       Returns a list of (code_line, label).
    """
    val1 = random.randint(10, 50)
    val2 = random.randint(51, 100)
    return [
        (f"    {var_name} = {val1}  # Dead assignment", 1), # Dead
        (f"    {var_name} = {val2}  # Live assignment (shadows previous one)", 0) # Alive
    ]

def generate_dead_blocks_and_calls() -> List[Tuple[str, int]]:
    """Generates unreachable code blocks and unused calls/expressions.
       Returns a list of (code_line, label).
    """
    dead_code = []
    
    # Unreachable block: Keep the condition (0/Alive), eliminate the body (1/Dead)
    # FIX: Correctly label the conditional line (if 1 == 0:) as 0 (Alive/Keep).
    dead_code.append((f"    if 1 == 0:  # Trivial false condition", 0)) 
    dead_code.append((f"        dead_call = random.random()  # Unreachable function call", 1))
    dead_code.append((f"        print('This is impossible')", 1))
    
    # Redundant expression/call not assigned or used
    if random.random() < 0.5:
        dead_code.append((f"    {''.join(random.choices('abcdef', k=5))} * {random.randint(2, 5)} # Redundant expression", 1))
        
    # Dead function call (assuming random is imported globally or is a simple built-in)
    dead_code.append((f"    {random.choice(['time.sleep', 'print'])}('Debugging code') # Dead print/call", 1))

    return dead_code

def create_dataset() -> List[Dict]:
    """Creates a dataset by injecting dead code into live functions."""
    dataset = []
    for i in range(NUM_SAMPLES):
        # 1. Select a base function
        base_code, alive_parts = random.choice(BASE_FUNCTIONS)
        
        lines = base_code.strip().split('\n')
        labels = [0] * len(lines)
        
        # 2. Generate Dead Code Patterns
        dead_vars = generate_dead_assignments("unused_var_", random.randint(1, 3))
        dead_blocks_labels = generate_dead_blocks_and_calls()
        shadowed_code_labels = generate_shadowed_assignment("temp_val")
        
        # 3. Inject Dead Code (randomly across the code)
        
        # Inject pure unused variables
        insert_line = random.randint(1, len(lines))
        for var_assignment in dead_vars:
            lines.insert(insert_line, var_assignment)
            labels.insert(insert_line, 1) # Label as DEAD
            insert_line += 1
            
        # Inject shadowed assignment (first line is dead, second is live)
        insert_line = random.randint(1, len(lines))
        for j, (line, label) in enumerate(shadowed_code_labels):
            lines.insert(insert_line + j, line)
            labels.insert(insert_line + j, label)
        
        # Inject unreachable/redundant code at the end
        for line, label in dead_blocks_labels:
            lines.append(line)
            labels.append(label)
        
        # 4. Format for the dataset
        full_context = "\n".join(lines).strip()
        for line, label in zip(lines, labels):
            # Only save lines with actual content for training efficiency
            stripped_line = line.strip()
            if stripped_line:
                dataset.append({
                    "code_snippet": stripped_line,
                    "label": label,  # 0: Alive, 1: Dead
                    "full_context": full_context
                })
            
    return dataset

if __name__ == "__main__":
    synthetic_data = create_dataset()
    
    # Use JSON Lines format (lines=True) which is better for large datasets
    with open(OUTPUT_FILE, 'w') as f:
        # We write records line by line to ensure 'lines=True' format is maintained
        for item in synthetic_data:
            f.write(json.dumps(item) + '\n')
        
    print(f"--- Dataset Generation Complete ---")
    print(f"Generated {len(synthetic_data)} lines of labeled code snippets.")
    print(f"Saved to {OUTPUT_FILE} (JSON Lines format).")
    print("\nExample Data Point:")
    print(json.dumps(synthetic_data[random.randint(0, len(synthetic_data) - 1)], indent=4))