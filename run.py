#!/usr/bin/env python3
"""CLI entry point (Task 5/6).

    python run.py --n 8 --K 3 --density 0.3 --seed 1
    python run.py --input instance.json --output result.json
"""
import argparse
import json
import sys

from src.io_utils import instance_from_generator, load_instance, result_to_dict, save_result
from src.scheduler import solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int)
    ap.add_argument("--K", type=int)
    ap.add_argument("--density", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--input", type=str, help="load instance JSON instead of generating")
    ap.add_argument("--output", type=str, help="write result JSON here (default: stdout)")
    ap.add_argument("--local-search-c", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args()

    if args.input:
        instance = load_instance(args.input)
    elif args.n and args.K:
        instance = instance_from_generator(args.n, args.K, args.density, args.seed, d=args.d)
    else:
        print("Provide either --input FILE or --n N --K K", file=sys.stderr)
        sys.exit(1)

    result = solve(instance, local_search_c=args.local_search_c, time_budget_s=args.time_budget)

    if args.output:
        save_result(result, args.output)
    else:
        print(json.dumps(result_to_dict(result), indent=2))


if __name__ == "__main__":
    main()
