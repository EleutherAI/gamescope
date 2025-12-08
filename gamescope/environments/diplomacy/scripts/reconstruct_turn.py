import sys
import os
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Resolve repo root from this file's location
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]  # gamescope/environments/diplomacy/scripts -> root

# Add relevant paths to sys.path
if str(_REPO_ROOT / "gamescope/environments/diplomacy/vendor/diplobench") not in sys.path:
    sys.path.append(str(_REPO_ROOT / "gamescope/environments/diplomacy/vendor/diplobench"))

if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

try:
    from diplomacy_game.environment import DiplomacyEnvironment
    from diplomacy_game.agent import LLMAgent, ENGINE_TO_CODE, CODE_TO_ENGINE, GENERAL_PLAY_TIPS
except ImportError as e:
    print(f"Error: Could not import diplomacy_game modules. {e}")
    # Print sys.path for debugging
    print("sys.path:", sys.path)
    sys.exit(1)

# Configure logging to avoid spam
logging.basicConfig(level=logging.ERROR)

def reconstruct_prompt(log_path: str, power: str, phase_name: str, subround: int = None, task_type: str = None) -> str:
    # Load Log
    with open(log_path, 'r') as f:
        data = json.load(f)

    turn_history = data['turn_history']
    negotiation_history = data.get('negotiation_history', [])
    
    # Find index for phase
    phase_idx = -1
    for i, entry in enumerate(turn_history):
        if entry['phase'] == phase_name:
            phase_idx = i
            break
    
    if phase_idx == -1:
        error_msg = f"Error: Phase {phase_name} not found in turn_history."
        print(error_msg)
        return ""

    target_phase_data = turn_history[phase_idx]
    
    # 1. Environment State & Adjudication (The "One-Step Back" Trick)
    # If not the first turn, load N-1 and step forward to generate outcomes text
    map_path = _REPO_ROOT / "gamescope/environments/diplomacy/maps/treaty_of_verdun.map"
    if not map_path.exists():
        # Fallback to standard map
        map_path = 'standard'
    else:
        map_path = str(map_path)
            
    env = DiplomacyEnvironment(map_name_or_path=map_path)
    
    if phase_idx > 0:
        prev_data = turn_history[phase_idx - 1]
        prev_state = prev_data['env_state']
        accepted_orders = prev_data['accepted_orders']
        
        # Load N-1
        env.game.set_state(prev_state)
        
        # Apply orders
        for pwr, orders in accepted_orders.items():
            if orders:
                env.set_orders(pwr, orders)
        
        # Step
        env.step()
    else:
        # First turn, just load state
        env.game.set_state(target_phase_data['env_state'])
    
    # 2. Agent State
    agents_data = data['agents_data']
    if power not in agents_data:
        print(f"Error: Power {power} not found in agents_data.")
        return ""
        
    agent_info = agents_data[power]
    personality = agent_info['personality']
    goals = agent_info.get('goals', [])
    
    # Journal Slicing
    full_journal = agent_info['journal']
    sliced_journal = []
    
    for entry in full_journal:
        # Check if entry starts with current phase
        if entry.strip().startswith(phase_name):
            break
        sliced_journal.append(entry)
        
    # Relationships
    # "Load player_relationships from the same turn_history entry"
    relationships = target_phase_data.get('player_relationships', {}).get(power, {})
    
    # Instantiate Agent
    agent = LLMAgent(power, personality, goals, journal=sliced_journal)
    agent.relationships = relationships
    
    # Get Observation
    observation = env.get_observation_for_power(power)
    
    # Inject missing fields required by prompt builder
    if "rl_recommendations" not in observation:
        observation["rl_recommendations"] = {power: "None available."}
    
    # 3. Prompt Reconstruction
    
    # Detect task type if not provided
    if task_type is None:
        if subround is not None:
            task_type = "negotiation"
        else:
            task_type = "orders"

    if task_type == "negotiation":
        # Negotiation Prompt
        
        # Locate negotiation history for this phase
        neg_entry = None
        for entry in negotiation_history:
            if entry['phase'] == phase_name:
                neg_entry = entry
                break
        
        inbox_history = []
        if neg_entry:
             raw_subrounds = neg_entry.get('subrounds', [])
             for sr in raw_subrounds:
                 if sr['subround_index'] < subround:
                     # Transform to format expected by format_inbox_history
                     sent = [msg for msg in sr['sent_missives'] if msg['sender'] == power]
                     received = sr['received_missives'].get(power, [])
                     inbox_history.append({
                         "subround_index": sr['subround_index'],
                         "sent_missives": sent,
                         "received_missives": received
                     })

        formatted_inbox = agent.format_inbox_history(inbox_history)
        prompt = generate_negotiation_prompt(agent, observation, subround, formatted_inbox)
        return prompt

    elif task_type == "journal":
        # Journal / Summary Prompt
        
        # Locate negotiation history for this phase
        neg_entry = None
        for entry in negotiation_history:
            if entry['phase'] == phase_name:
                neg_entry = entry
                break
        
        inbox_history = []
        if neg_entry:
             # For summary, we want the FULL history of this phase
             raw_subrounds = neg_entry.get('subrounds', [])
             for sr in raw_subrounds:
                 sent = [msg for msg in sr['sent_missives'] if msg['sender'] == power]
                 received = sr['received_missives'].get(power, [])
                 inbox_history.append({
                     "subround_index": sr['subround_index'],
                     "sent_missives": sent,
                     "received_missives": received
                 })

        formatted_inbox = agent.format_inbox_history(inbox_history)
        prompt = generate_summary_prompt(agent, observation, formatted_inbox)
        return prompt
        
    else:
        # Orders Prompt
        
        # Inject full negotiation history for this phase if available
        neg_entry = None
        for entry in negotiation_history:
            if entry['phase'] == phase_name:
                neg_entry = entry
                break
        
        if neg_entry:
            # Reconstruct per-agent inbox history from subrounds
            inbox_history = []
            raw_subrounds = neg_entry.get('subrounds', [])
            for sr in raw_subrounds:
                sub_i = sr['subround_index']
                # Filter sent/received for THIS power
                sent = [msg for msg in sr['sent_missives'] if msg['sender'] == power]
                received = sr['received_missives'].get(power, [])
                
                if sent or received:
                    inbox_history.append({
                        "subround_index": sub_i,
                        "sent_missives": sent,
                        "received_missives": received
                    })
            
            observation['negotiation_history'] = inbox_history

        prompt_only = agent.build_prompt_orders_only(observation)
        
        # System prompt logic
        phase_type = phase_name[-1] if phase_name else "?"
        base_system = (
            "You are an AI playing Diplomacy. Play faithfully to your personality. "
            "Always try to move the game forward and avoid stalemate per your objectives. "
            "Use 3-letter power codes (do not reference powers that do not exist on the current map). "
            "Output valid JSON with exactly two fields in this order: 'reasoning' (list of strings) and 'orders' (list of strings)."
        )
        
        if phase_type == "A":
            system_text = base_system + (
                "\nThis is an ADJUSTMENT phase. You can ONLY issue these types of orders:"
                "\n- Build: 'A/F <province> B'"
                "\n- Disband: 'A/F <province> D'"
                "\nExample adjustment orders: 'A PAR B', 'F BRE B', 'A MUN D'"
            )
        else:
            system_text = base_system
            
        full_prompt = f"SYSTEM:\n{system_text}\n\nUSER:\n{prompt_only}"
        return full_prompt

def generate_negotiation_prompt(agent, observation, sub_round_index, formatted_inbox_history):
    final_round_note = ''
    if sub_round_index == agent.NUM_MISSIVES:
        final_round_note = '''
    * This is the final round of missives this phase. Missives will be one-way and you will not get a response.'''

    formatted_journal = "\n".join(agent._format_journal(agent.journal[-15:], observation["phase"]))
    
    tips = GENERAL_PLAY_TIPS + f"\n- Game ends at {1900 + agent.max_turns}."

    rl_recs = observation.get("rl_recommendations", {}).get(agent.power_name, "None available.")

    misinformation = "" # Reconstructed prompt assumes no random misinformation

    prompt_text = f"""
=== YOUR POWER ===
Your Nation: {agent.power_name}
Personality: {agent.personality}

=== TIPS TO WIN AT DIPLOMACY ===
{tips}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== STRATEGIC OVERVIEW ===
{observation["strategic_overview"]}

=== ENGINE SUGGESTED MOVES ===
IMPORTANT: These may or may not not align with your diplomatic goals. Feel free to use or ignore them at your discretion.
{rl_recs}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== LAST PHASE OUTCOMES ===
{observation.get("last_turn_outcomes", '')}

=== RECENT PRIVATE JOURNAL ===
{formatted_journal}

=== RELATIONSHIPS ===
{agent._relationship_dump()}

=== COMMUNICATION THIS PHASE ===
{formatted_inbox_history}
{misinformation}
=== INSTRUCTIONS ===
There are {agent.NUM_MISSIVES} rounds of communications this turn. This missive will be {sub_round_index} of {agent.NUM_MISSIVES}.
You can send up to 3 short missives in this round of communications, each to one or multiple recipients.
Use 3-letter codes to designate recipients.
Special recipient 'ALL' means broadcast to everyone.

Convoy Rules:
- If convoying with another player, you must both negotiate and verify the move is valid & listed in strategic overview
- For convoying your own units, issue orders for both the fleet and the army being convoyed
- Chain convoys ARE possible but must be determined from game state - only single convoys shown in overview

Tips:
- Other than diplomacy, this is the time to coordinate specific attacks and defensive maneuvers with other powers.
- Diplomacy isn't just words, it's about backing your commitments with specific actions. It's about what you can offer and what you can extract.
- Move the game forward and avoid stalemate.

Return valid JSON with a 'missives' list containing up to 3 missives, each with:
- 'recipients': list of 3-letter codes, you may list multiple recipients
- 'body': string (keep to 1 paragraph)

Output format:
{{
    "missives": [
        {{
            "recipients": ["Recipient 1", ...],
            "body": "Your 1 paragraph message"
        }},
        ...
    ]
}}

=== PHASE & TIMING ===
Phase: {observation["phase"]}
Negotiation Round: {sub_round_index} of {agent.NUM_MISSIVES}{final_round_note}

No extra commentary in response.

/no_think

Compose your missives now according to the instructions.
"""

    system_text = (
        "You are an AI Diplomacy player. This is the negotiation stage where missives can be sent to further your interests. "
        "Play faithfully to your personality profile. Always try to move the game forward per your objectives. "
        "Only output valid JSON with the key 'missives'. Use 3-letter codes for powers; 'ALL' broadcasts to everyone."
        "/no_think"
    )

    return f"SYSTEM:\n{system_text}\n\nUSER:\n{prompt_text}"

def generate_summary_prompt(agent, observation, formatted_inbox):
    prompt_text = f"""
=== YOUR POWER ===
Your Nation: {agent.power_name}
Personality: {agent.personality}

=== GAME STATE ===
{json.dumps(observation.get("board_state", {}), indent=2)}

=== RECENT MOVES ===
{json.dumps(observation.get("recent_moves", {}), indent=2)}

=== THIS ROUND'S NEGOTIATION HISTORY ===
{formatted_inbox}

=== YOUR RELATIONSHIPS ===
{agent._relationship_dump()}

=== INSTRUCTIONS ===
Summarize the negotiations for your private journal.
- Summarise the events of the prior move.
- Briefly note the important happenings of the negotiations.
- Make sure to note any promises made or broken, by yourself and by others.
- Log your intended plans, coordinations, and deceptions.
- Log your intended moves.
- Keep the summary concise but specific & informative.

Return the summary in valid JSON format with the following structure:
{{
    "prior_move_summary": "summary of the prior move",
    "negotiation_summary": "summary of negotiations",
    "intent": "specific intents, plans and moves",
    "rship_updates": [list of any changed relationships in the format 'ENG-FRA+', 'ENG-RUS--', etc.]
}}

Use only 3-letter codes for powers.

/no_think

=== PHASE & TIMING ===
Phase: {observation["phase"]}

Compose your summary now according to the instructions.
"""

    system_text = (
        "You are an AI Diplomacy player concluding negotiations. Play faithfully to your personality profile. "
        "Always try to move the game forward per your objectives. Avoid repetition in your journal. "
        "Return valid JSON with 'prior_move_summary', 'negotiation_summary', 'intent', 'rship_updates'. Use 3-letter codes for powers."
        "/no_think"
    )

    return f"SYSTEM:\n{system_text}\n\nUSER:\n{prompt_text}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--power", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--subround", type=int, default=None)
    
    args = parser.parse_args()
    
    prompt = reconstruct_prompt(args.log, args.power, args.phase, args.subround)
    print(prompt)

