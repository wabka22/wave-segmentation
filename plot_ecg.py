import wfdb
import matplotlib.pyplot as plt

record = wfdb.rdrecord("data/ludb/data/1")
ann = wfdb.rdann("data/ludb/data/1", "ii")

signal = record.p_signal[:,1]

plt.figure(figsize=(12,4))
plt.plot(signal)

for s in ann.sample:
    plt.axvline(s, color='red', alpha=0.3)

plt.title("ECG with annotations")
plt.show()
