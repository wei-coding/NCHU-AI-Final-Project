import csv

def load_logs(filename='logs.csv'):
    record = 0
    n_games = 0
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = max(record, int(row['record']))
            n_games = max(n_games, int(row['n_games']))
    return record, n_games


def save_logs(record, score, n_games, filename='logs.csv', init=False):
    if init:
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n_games', 'score', 'record'])
            writer.writeheader()
    else:
        with open(filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n_games', 'score', 'record'])
            writer.writerow({'record': record, 'score': score, 'n_games': n_games})