import json
import random

DUMMY_TEXT = """
### Commit Message
update
### Code Difference
"""
with open('sample_diff.txt', 'r') as f:
    DUMMY_TEXT += f.read()

data = {}
for project in ['foo', 'bar', 'hello', 'world', 'this', 'is', 'the', 'cat']:
    commits = []
    for i in range(1000):
        commits.append({
            'text': DUMMY_TEXT,
            'is_buggy': 1 if random.random() < 0.1 else 0,
            'index': i,
            'reported_index': i + random.randint(2, 30),
        })
    data[project] = commits
    
with open('sample_data.json', 'w') as f:
    json.dump(data, f, indent=1)