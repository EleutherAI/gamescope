from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is importable so `gamescope.libs.*` works even if cwd changes
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from dotenv import dotenv_values  # noqa: E402

from gamescope.libs.run_utils import run_context  # noqa: E402


# Mapping from convenient short names (or legacy names) to what diplobench expects
ENV_VAR_MAPPINGS = {
    "DEFAULT_MODEL": "DEFAULT_AGENT_MODEL",
    "TARGET_MODEL": "TARGET_AGENT_MODEL",
    "DEFAULT_BASE_URL": "DEFAULT_AGENT_BASE_URL",
    "TARGET_BASE_URL": "TARGET_AGENT_BASE_URL",
    "DEFAULT_API_KEY": "DEFAULT_AGENT_API_KEY",
    "TARGET_API_KEY": "TARGET_AGENT_API_KEY",
    "DEFAULT_TIMEOUT": "DEFAULT_AGENT_TIMEOUT_SECONDS",
    "TARGET_TIMEOUT": "TARGET_AGENT_TIMEOUT_SECONDS",
}


def _build_subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    
    # Apply mappings from current environment (CLI args take precedence)
    # If the user specified a short name (e.g. DEFAULT_MODEL), forward it to the long name
    # that diplobench expects (DEFAULT_AGENT_MODEL), overriding any existing value there
    # to ensure the CLI arg wins over .env or other defaults.
    for short_name, long_name in ENV_VAR_MAPPINGS.items():
        if short_name in env:
            env[long_name] = env[short_name]

    dotenv_path = _REPO_ROOT / ".env"
    if dotenv_path.exists():
        for key, value in dotenv_values(dotenv_path).items():
            if value is None:
                continue
            env.setdefault(key, value)
    
    # Ensure Python imports work when running module with cwd set to artifacts dir
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if str(_REPO_ROOT) not in parts:
        parts.insert(0, str(_REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _final_centers_from_env_state(env_state: dict) -> Dict[str, int]:
    centers = env_state.get("centers", {})
    return {str(k): int(len(v or [])) for k, v in centers.items()}


def _resolve_to_abs_if_exists(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to current working dir
    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return str(cand)
    # Try relative to repo root
    cand2 = (_REPO_ROOT / p).resolve()
    if cand2.exists():
        return str(cand2)
    # Fallback: return original string (let downstream handle names like 'standard')
    return path_str


def _aggregate(per_game: List[dict]) -> dict:
    # Average centers per power and winner distribution
    sum_centers: Dict[str, int] = defaultdict(int)
    count_centers: Counter = Counter()
    winners: Counter = Counter()
    successes = 0

    for rec in per_game:
        if rec.get("status") == "success":
            successes += 1
        centers = rec.get("final_centers", {})
        for pwr, n in centers.items():
            sum_centers[pwr] += int(n)
            count_centers[pwr] += 1
        for w in rec.get("winners", []):
            winners[w] += 1

    avg_centers = {p: (sum_centers[p] / count_centers[p]) for p in sum_centers.keys() if count_centers[p] > 0}
    return {
        "num_games": len(per_game),
        "num_success": int(successes),
        "avg_centers": avg_centers,
        "winner_counts": dict(winners),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Diplobench by running N games and logging final results.")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to run.")
    parser.add_argument("--turns", type=int, default=10, help="Max number of years to simulate (as in diplobench main).")
    parser.add_argument("--map", type=str, default=None, help="Map name or path to .map (defaults to standard).")
    parser.add_argument("--no_negotiate", action="store_true", help="Disable negotiation phase (default is negotiate on).")
    parser.add_argument("--enable_probe", action="store_true", help="Enable deception probe during negotiation.")
    parser.add_argument("--probe_artifact", type=str, default=None, help="Path to trained probe artifact (pickle).")
    parser.add_argument("--probe_power", type=str, default=None, help="Specific power code to probe/test (e.g., AUT, ENG, FRA).")
    parser.add_argument("--probe_random", action="store_true", help="Randomize which power is probed/tested.")
    parser.add_argument("--probe_device", type=str, default="auto", help="Device for probe model: 'auto' (multi-GPU), 'cuda:N' (specific GPU), or 'cpu'. Default: auto")
    parser.add_argument("--probe_direction", type=str, choices=["player", "opponents"], default="player", help="Probe 'player' (target's own messages) or 'opponents' (all others, report to target only).")
    parser.add_argument("--random_probe_outputs", action="store_true", help="Use random probe outputs (uniform [0,1]) instead of actual probe predictions. Useful for testing/benchmarking.")
    parser.add_argument("--results_dir", type=str, default=str(_REPO_ROOT / "results" / "diplomacy"), help="Base results directory.")
    parser.add_argument("--experiment_name", type=str, default="diplobench_eval", help="Experiment subdirectory under results.")
    args = parser.parse_args()

    base_dir = Path(args.results_dir) / args.experiment_name
    env = _build_subprocess_env()

    # Run context sets up run_dir, config, metadata, and status tracking
    with run_context(base_dir=base_dir, run_prefix="diplobench_eval", config_args=vars(args)) as run_dir:
        run_dir = Path(run_dir)
        artifacts_dir = run_dir / "artifacts"
        logs_dir = run_dir / "logs"
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Per-game JSONL metrics
        per_game_jsonl = metrics_dir / "games.jsonl"
        per_game_records: List[dict] = []

        for i in range(int(args.num_games)):
            game_id = f"game_{i:03d}-{uuid.uuid4().hex[:8]}"
            log_path = logs_dir / f"{game_id}.log"

            cmd: List[str] = [
                sys.executable,
                "-m",
                "gamescope.environments.diplomacy.vendor.diplobench.main",
                "--game-id",
                game_id,
                "--turns",
                str(int(args.turns)),
            ]
            if not args.no_negotiate:
                cmd.append("--negotiate")
            if args.map:
                map_arg = _resolve_to_abs_if_exists(str(args.map))
                cmd.extend(["--map", str(map_arg)])
            if args.probe_power:
                cmd.extend(["--probe-power", str(args.probe_power)])
            if args.probe_random:
                cmd.append("--probe-random")
            if args.enable_probe:
                cmd.append("--enable-probe")
                if args.probe_artifact:
                    probe_arg = _resolve_to_abs_if_exists(str(args.probe_artifact))
                    cmd.extend(["--probe-artifact", str(probe_arg)])
                # Pass probe device configuration
                if args.probe_device:
                    cmd.extend(["--probe-device", str(args.probe_device)])
            # Always pass direction explicitly so main sees the intended value
            cmd.extend(["--probe-direction", str(args.probe_direction)])
            if args.random_probe_outputs:
                cmd.append("--random-probe-outputs")

            # Run diplobench main; set cwd to artifacts so <game_id>.json lands in artifacts
            with open(log_path, "w") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(artifacts_dir),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            status = "success" if proc.returncode == 0 else "failed"
            json_path = artifacts_dir / f"{game_id}.json"

            record = {
                "game_id": game_id,
                "status": status,
                "return_code": int(proc.returncode),
                "json_artifact": str(json_path),
                "map": args.map,
                "turns": int(args.turns),
                "negotiate": (not args.no_negotiate),
                "enable_probe": bool(args.enable_probe),
                "probe_artifact": args.probe_artifact,
                "probe_device": args.probe_device,
                "probe_direction": args.probe_direction,
                "random_probe_outputs": bool(args.random_probe_outputs),
            }

            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    env_state = data.get("env_state", {})
                    final_centers = _final_centers_from_env_state(env_state)
                    record["final_centers"] = final_centers
                    # Winner(s): engine power names with max centers
                    if final_centers:
                        max_sc = max(final_centers.values())
                        winners = [k for k, v in final_centers.items() if int(v) == int(max_sc)]
                        record["winners"] = winners
                        record["max_centers"] = int(max_sc)
                except Exception:
                    # Keep record minimal if parse fails
                    pass

            with open(per_game_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record) + "\n")
            per_game_records.append(record)

        summary = _aggregate(per_game_records)
        summary_path = metrics_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        # Also print a brief summary to stdout
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


