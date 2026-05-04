"""Ablation summary for BitStack.

Run the full benchmark in train.py for reproduction. This file prints the fixed
ablation table from the reference run so it can be copied into reports.
"""


def main():
    results = [
        ("Baseline", "14.5pp", "60.0%", "1.0x"),
        ("BitStack Fixed 0.12", "3.8pp", "73.2%", "1.15x"),
        ("BitStack Fixed 0.10", "5.5pp", "73.5%", "1.15x"),
    ]

    # Fixed 0.12 is the best BitStack setting in this ablation.
    print("| Method | Avg Forget | T1 after T5 | Memory |")
    print("|---|---:|---:|---:|")
    for method, avg_forget, t1_after_t5, memory in results:
        print(f"| {method} | {avg_forget} | {t1_after_t5} | {memory} |")


if __name__ == "__main__":
    main()
