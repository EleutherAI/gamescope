import re
import json
import sys
from collections import defaultdict

def parse_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    phases = []
    current_phase = None
    
    # Regex patterns
    phase_start_re = re.compile(r"=== PHASE: (\w+) ===")
    
    # We'll just grab the whole block of messages/orders for manual scanning
    # This is a simplified version of the previous script
    
    lines = content.split('\n')
    iterator = iter(lines)
    
    current_phase = {'name': 'START', 'content': []}
    
    for line in iterator:
        phase_match = phase_start_re.search(line)
        if phase_match:
            phases.append(current_phase)
            current_phase = {'name': phase_match.group(1), 'content': []}
        
        if current_phase:
            current_phase['content'].append(line)
            
    phases.append(current_phase)
    return phases

def extract_dialogue_and_orders(phases):
    for phase in phases:
        if phase['name'] == 'START': continue
        
        print(f"## Phase {phase['name']}")
        
        content = "\n".join(phase['content'])
        
        # Extract messages
        # Look for the message blocks
        # --------------
        # PLAYER
        # [{'recipients': ...}]
        
        # We can just grep for the list of dicts structure or the player headers
        
        # Let's just print the whole phase content but filtered for relevant lines
        # to save context
        
        lines = phase['content']
        for i, line in enumerate(lines):
            if "INFO:__main__:Orders from" in line:
                print(f"ORDERS: {line.strip()}")
            elif "INFO:__main__:Setting validated orders" in line:
                print(f"VALIDATED: {line.strip()}")
            elif line.strip().startswith("[{'recipients'"):
                # Find who sent it (line before)
                sender = lines[i-1].strip() if i > 0 else "UNKNOWN"
                if sender in ['EAS', 'MID', 'WES']:
                    print(f"MSG from {sender}: {line.strip()}")
        
        print("-" * 20)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scan_log.py <log_file>")
        sys.exit(1)
        
    log_path = sys.argv[1]
    phases = parse_log(log_path)
    extract_dialogue_and_orders(phases)

