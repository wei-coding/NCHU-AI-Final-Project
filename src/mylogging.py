import csv
import os

def load_logs(filename='logs.csv'):
    if os.path.exists(filename):
        print('found log, restoring...')
        record = 0
        n_games = 0
        with open(filename) as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = max(record, int(row['record']))
                n_games = max(n_games, int(row['n_games']))
        print(n_games, record)
        return record, n_games
    else:
        return None, None


def save_logs(record=0, score=0, n_games=0, filename='logs.csv'):
    if os.path.exists(filename):
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n_games', 'score', 'record'])
            writer.writerow({'record': record, 'score': score, 'n_games': n_games})
    else:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n_games', 'score', 'record'])
            writer.writeheader()
            writer.writerow({'record': record, 'score': score, 'n_games': n_games})