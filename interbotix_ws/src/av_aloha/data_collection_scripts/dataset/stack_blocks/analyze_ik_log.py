import numpy as np

LOG_PATH = "timing_episode_0000.npz"


def print_stats(name, arr):
    if arr.size == 0:
        print(f"{name}: no data")
        return
    print(
        f"{name}: n={arr.size}, mean={arr.mean()*1000:.2f} ms, "
        f"std={arr.std()*1000:.2f} ms, min={arr.min()*1000:.2f} ms, max={arr.max()*1000:.2f} ms"
    )


def main():
    data = np.load(LOG_PATH)

    # Arrays are in seconds (from your code), so convert to ms when printing
    loop = data["loop"]
    ik_solve = data["ik_solve"]
    ik_section = data["ik_section"]
    cmd = data["cmd"]
    key = data["key"]
    headset = data["headset"]
    log = data["log"]

    print("=== Loop timing ===")
    print_stats("loop", loop)

    print("\n=== IK solve timing (solver only) ===")
    print_stats("ik_solve", ik_solve)

    print("\n=== IK section timing (whole IK section) ===")
    print_stats("ik_section", ik_section)

    print("\n=== Command send timing ===")
    print_stats("cmd", cmd)

    print("\n=== Other sections ===")
    print_stats("key", key)
    print_stats("headset", headset)
    print_stats("log", log)

    # Quick sanity: show a few samples
    print("\nSample ik_solve times (ms):", (ik_solve[:10] * 1000))
    print("Sample ik_section times (ms):", (ik_section[:10] * 1000))
    print("Sample loop times (ms):", (loop[:10] * 1000))


if __name__ == "__main__":
    main()