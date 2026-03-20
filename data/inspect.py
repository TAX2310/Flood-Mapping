from collections import Counter

event_counts = Counter()

for sample in pairs:
    event_id = sample["event_id"]
    event_counts[event_id] += 1

for event_id, count in sorted(event_counts.items(), key=lambda x: x[1], reverse=True):
    print(event_id, count)