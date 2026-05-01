from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

ea = EventAccumulator("logs/rsl_rl/anymal_c_rough/2026-02-03_11-13-59/events.out.tfevents.1770135247.isengard.2797158.0")
ea.Reload()

# List available scalar tags
print(ea.Tags()["scalars"])

# Read a tag
events = ea.Scalars("Train/mean_reward")
steps  = np.array([e.step  for e in events])
values = np.array([e.value for e in events])

# Numpy computations
mean   = np.mean(values)
std    = np.std(values)
smooth = np.convolve(values, np.ones(20)/20, mode='valid')   # moving average
print(f"Mean: {mean}, Std: {std}, Smooth: {smooth}")