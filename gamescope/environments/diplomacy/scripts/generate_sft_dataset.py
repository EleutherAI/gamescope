import os
import sys
import json
import logging
import argparse
from glob import glob
from tqdm import tqdm
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
    from gamescope.environments.diplomacy.scripts.reconstruct_turn import reconstruct_prompt
    from diplomacy_game.llm import generate
    from gamescope.libs.run_utils import run_context
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_log(log_path, output_file, model_name, subrounds_per_phase=3, dry_run=False):
    """
    Iterates through a game log, reconstructs prompts for all powers and phases,
    queries the teacher model, and appends results to output_file.
    """
    
    with open(log_path, 'r') as f:
        data = json.load(f)
        
    game_id = data.get('game_id', 'unknown')
    turn_history = data['turn_history']
    negotiation_history = data.get('negotiation_history', [])
    
    # Map phase to negotiation history for quick lookup
    neg_map = {entry['phase']: entry for entry in negotiation_history}
    
    # Powers present in the game
    powers = list(data.get('agents_data', {}).keys())
    
    results_buffer = [] # Deprecated, we write immediately now
    
    for turn in tqdm(turn_history, desc=f"Processing {os.path.basename(log_path)}"):
        phase = turn['phase']
        
        # 1. Orders Phase Prompts
        for power in powers:
            # Reconstruct Orders Prompt
            # Subround None means orders
            try:
                prompt = reconstruct_prompt(log_path, power, phase, subround=None)
                if not prompt:
                    continue
                
                # Split System and User for API
                # Format is "SYSTEM:\n...\n\nUSER:\n..."
                parts = prompt.split("USER:\n")
                if len(parts) == 2:
                    system_text = parts[0].replace("SYSTEM:\n", "").strip()
                    user_text = parts[1].strip()
                else:
                    system_text = ""
                    user_text = prompt
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would generate orders for {game_id} {phase} {power}")
                    completion = "DRY_RUN_COMPLETION"
                else:
                    # Generate Completion
                    logger.info(f"Generating orders completion for {game_id} {phase} {power}")
                    completion = generate(
                        prompt_text=user_text,
                        system_text=system_text,
                        model_name=model_name,
                        temperature=0.7 
                    )
                
                if completion:
                    record = {
                        "game_id": game_id,
                        "phase": phase,
                        "power": power,
                        "type": "orders",
                        "prompt": prompt,
                        "completion": completion,
                        "model": model_name
                    }
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(record) + "\n")
                    
            except Exception as e:
                logger.error(f"Error processing {phase} {power} orders: {e}")

        # 2. Negotiation Phase Prompts
        # Only if this phase had negotiations
        if phase in neg_map:
            neg_entry = neg_map[phase]
            # Negotiation usually has multiple subrounds.
            # We want to reconstruct prompt for each subround that actually happened.
            # neg_entry['subrounds'] is list of subrounds.
            
            for subround_data in neg_entry.get('subrounds', []):
                subround_idx = subround_data['subround_index']
                
                # Check if we should process this subround
                # Maybe user wants all subrounds? "per negotiation sub-round"
                
                for power in powers:
                    try:
                        prompt = reconstruct_prompt(log_path, power, phase, subround=subround_idx)
                        if not prompt:
                            continue

                        parts = prompt.split("USER:\n")
                        if len(parts) == 2:
                            system_text = parts[0].replace("SYSTEM:\n", "").strip()
                            user_text = parts[1].strip()
                        else:
                            system_text = ""
                            user_text = prompt
                        
                        if dry_run:
                            logger.info(f"[DRY RUN] Would generate negotiation for {game_id} {phase} {power} round {subround_idx}")
                            completion = "DRY_RUN_COMPLETION"
                        else:
                            logger.info(f"Generating negotiation completion for {game_id} {phase} {power} round {subround_idx}")
                            completion = generate(
                                prompt_text=user_text,
                                system_text=system_text,
                                model_name=model_name,
                                temperature=0.7
                            )
                        
                        if completion:
                            record = {
                                "game_id": game_id,
                                "phase": phase,
                                "power": power,
                                "type": "negotiation",
                                "subround": subround_idx,
                                "prompt": prompt,
                                "completion": completion,
                                "model": model_name
                            }
                            with open(output_file, 'a') as f:
                                f.write(json.dumps(record) + "\n")

                    except Exception as e:
                        logger.error(f"Error processing {phase} {power} neg round {subround_idx}: {e}")

            # 3. Journal / Summary Phase Prompts (Post-Negotiation)
            for power in powers:
                try:
                    prompt = reconstruct_prompt(log_path, power, phase, task_type="journal")
                    if not prompt:
                        continue

                    parts = prompt.split("USER:\n")
                    if len(parts) == 2:
                        system_text = parts[0].replace("SYSTEM:\n", "").strip()
                        user_text = parts[1].strip()
                    else:
                        system_text = ""
                        user_text = prompt
                    
                    if dry_run:
                        logger.info(f"[DRY RUN] Would generate journal summary for {game_id} {phase} {power}")
                        completion = "DRY_RUN_COMPLETION"
                    else:
                        logger.info(f"Generating journal summary completion for {game_id} {phase} {power}")
                        completion = generate(
                            prompt_text=user_text,
                            system_text=system_text,
                            model_name=model_name,
                            temperature=0.7
                        )
                    
                    if completion:
                        record = {
                            "game_id": game_id,
                            "phase": phase,
                            "power": power,
                            "type": "journal",
                            "prompt": prompt,
                            "completion": completion,
                            "model": model_name
                        }
                        with open(output_file, 'a') as f:
                            f.write(json.dumps(record) + "\n")

                except Exception as e:
                    logger.error(f"Error processing {phase} {power} journal summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing game json logs")
    # Replaced output_file with results_dir/experiment_name pattern for reproducible run_context
    parser.add_argument("--results_dir", type=str, default="results/diplomacy/sft", help="Base results directory")
    parser.add_argument("--experiment_name", type=str, default="dataset_gen", help="Experiment name/subdirectory")
    parser.add_argument("--run_prefix", type=str, default="sft_run", help="Prefix for run directory")
    parser.add_argument("--model", required=True, help="Teacher model name (e.g. openai/gpt-4o)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of logs to process")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without calling LLM")
    
    args = parser.parse_args()
    
    log_files = glob(os.path.join(args.input_dir, "game_*.json"))
    if args.limit:
        log_files = log_files[:args.limit]
        
    print(f"Found {len(log_files)} log files.")
    
    base_dir = Path(args.results_dir) / args.experiment_name
    
    # Use run_context for experiment reproducibility
    with run_context(base_dir=base_dir, run_prefix=args.run_prefix, config_args=vars(args)) as run_dir:
        run_dir = Path(run_dir)
        output_file = run_dir / "sft_dataset.jsonl"
        logger.info(f"Writing dataset to {output_file}")
        
        for log_file in log_files:
            try:
                process_log(log_file, str(output_file), args.model, dry_run=args.dry_run)
            except Exception as e:
                logger.error(f"Failed to process {log_file}: {e}")



