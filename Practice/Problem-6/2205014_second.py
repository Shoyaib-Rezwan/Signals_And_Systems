import importlib

first_part=importlib.import_module("2205014_first")

Signal=first_part.Signal
LTI_System=first_part.LTI_System

# Assuming the Signal and LTI_System classes are imported or in the same file
def solve_part_two():
    # 1. Read the signal file and construct a Signal object [cite: 44]
    with open('input_signal.txt', 'r') as f:
        lines = f.readlines()
        # Line 1: start and end indices [cite: 40, 41]
        n_start, n_end = map(int, lines[0].split())
        # Line 2: signal values [cite: 42]
        values = list(map(float, lines[1].split()))

    noisy_signal = Signal(n_start, n_end)
    for i, val in enumerate(values):
        noisy_signal.set_value_at_time(n_start + i, val)

    # 2. Plot the noisy input signal [cite: 45, 54]
    noisy_signal.plot("Noisy Input Signal")

    # 3. Define 5-point moving average filter h(n) [cite: 46, 47]
    h_n = Signal(-2, 2)
    for n in range(-2, 3):
        h_n.set_value_at_time(n, 1/5)

    # 4. Create an LTI System using this impulse response [cite: 49]
    system = LTI_System(h_n)

    # 5. Compute the output signal [cite: 50]
    smoothed_signal = system.output(noisy_signal)

    # 6. Plot the smoothed output signal [cite: 51, 55]
    smoothed_signal.plot("Smoothed Output Signal")

if __name__ == "__main__":
    solve_part_two()