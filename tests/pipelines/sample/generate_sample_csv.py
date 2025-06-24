import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import random
import logging

log = logging.getLogger(__name__)

def create_sample_csv(num_rows=10):
    fake = Faker()
    categories = [
        "gas_transport", "shopping_pos", "shopping_net", "misc_pos", "entertainment",
        "home", "food_dining", "kids_pets", "misc_net", "health_fitness",
        "grocery_net", "grocery_pos", "travel"
    ]
    genders = ["M", "F"]

    data = []
    for _ in range(num_rows):
        row = {
            "cc_num": str(fake.credit_card_number()),
            "trans_num": fake.uuid4()[:8],
            "is_fraud": random.choice([0, 1]),
            "category": random.choice(categories),
            "gender": random.choice(genders),
            "merchant": fake.company(),
            "first": fake.first_name(),
            "last": fake.last_name(),
            "street": fake.street_address(),
            "city": fake.city(),
            "state": fake.state_abbr(),
            "zip": fake.zipcode(),
            "job": fake.job(),
            "merch_zipcode": fake.zipcode(),
            "amt": round(random.uniform(10, 1000), 2),
            "lat": round(fake.latitude(), 6),
            "long": round(fake.longitude(), 6),
            "city_pop": random.randint(10000, 1000000),
            "age": random.randint(18, 80),
            "merch_lat": round(fake.latitude(), 6),
            "merch_long": round(fake.longitude(), 6),
            "datetime": fake.iso8601(tzinfo=None, end_datetime=None),
        }
        data.append(row)

    df = pd.DataFrame(data)
    output_path = Path(__file__).resolve().parent / "sample.csv"
    df.to_csv(output_path, index=False)
    log.info(f"sample.csv with {num_rows} rows created at {output_path}")

if __name__ == "__main__":
    create_sample_csv()