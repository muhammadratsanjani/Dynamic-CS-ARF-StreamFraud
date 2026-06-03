import csv
import typing
from datetime import datetime

def iter_csv(file_path: str, target_col: str, convert_types: bool = True, limit: int = None) -> typing.Iterator[typing.Tuple[dict, int]]:
    """
    Generator function to read a CSV file line by line to simulate a data stream.
    Yields a tuple of (features: dict, label: int).
    """
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if limit and count >= limit:
                break
                
            label = int(row.pop(target_col))
            features = row
            
            if convert_types:
                for k, v in features.items():
                    try:
                        features[k] = float(v)
                    except ValueError:
                        pass # keep as string if it can't be converted
            
            yield features, label
            count += 1
