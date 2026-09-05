"""
WebLock - Synthetic Transaction Data Generator
------------------------------------------------
Generates a realistic e-commerce/payments transaction dataset with:
  - Normal, independent user behavior (majority class)
  - Coordinated FRAUD RINGS: groups of accounts sharing devices/IPs/cards,
    doing rapid-fire, similarly-structured transactions.

This lets us prove the core thesis of the project: fraud rings leave a
*relational* signature (shared entities) that row-level tabular models
struggle to see, but graph-based features/models can catch.

Output: data/transactions.csv
Columns:
  transaction_id, account_id, device_id, ip_id, card_id, merchant_id,
  amount, hour_of_day, day, is_weekend, merchant_category,
  time_since_last_txn_sec, txn_count_last_24h, is_fraud
"""

import numpy as np
import pandas as pd
import uuid
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RNG_SEED = 42
np.random.seed(RNG_SEED)
random.seed(RNG_SEED)

N_NORMAL_ACCOUNTS = 4000
N_FRAUD_RINGS = 15
RING_SIZE_RANGE = (4, 12)          # accounts per fraud ring
NORMAL_TXNS_PER_ACCOUNT = (1, 8)   # range of txns per normal account
FRAUD_TXNS_PER_ACCOUNT = (3, 15)   # fraud rings transact more densely
MERCHANT_CATEGORIES = [
    "electronics", "gift_cards", "travel", "groceries", "apparel",
    "digital_goods", "jewelry", "crypto_exchange", "money_transfer", "gaming"
]
# fraud rings disproportionately hit these "cash-out friendly" categories
FRAUD_PREFERRED_CATEGORIES = ["gift_cards", "crypto_exchange", "money_transfer", "digital_goods"]

def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:10]}"

rows = []

# ---------------------------------------------------------------
# 1. NORMAL ACCOUNTS: each account mostly has its OWN device/ip/card,
#    normal spending patterns, spread across time.
# ---------------------------------------------------------------
for i in range(N_NORMAL_ACCOUNTS):
    account_id = new_id("acct")
    device_id = new_id("dev")
    ip_id = new_id("ip")
    card_id = new_id("card")
    n_txns = np.random.randint(*NORMAL_TXNS_PER_ACCOUNT)

    last_txn_time = 0
    for _ in range(n_txns):
        hour = np.random.randint(0, 24)
        day = np.random.randint(1, 31)
        amount = round(np.random.lognormal(mean=3.7, sigma=1.1), 2)
        category = np.random.choice(MERCHANT_CATEGORIES)
        time_since_last = np.random.randint(600, 3600 * 72)  # minutes to days between txns
        txn_count_24h = np.random.randint(1, 6)

        rows.append({
            "transaction_id": new_id("txn"),
            "account_id": account_id,
            "device_id": device_id,
            "ip_id": ip_id,
            "card_id": card_id,
            "merchant_id": new_id("merch"),
            "amount": amount,
            "hour_of_day": hour,
            "day": day,
            "is_weekend": int(day % 7 in (0, 6)),
            "merchant_category": category,
            "time_since_last_txn_sec": time_since_last,
            "txn_count_last_24h": txn_count_24h,
            "is_fraud": 0
        })

# ---------------------------------------------------------------
# 2. FRAUD RINGS: groups of accounts sharing a SMALL pool of devices/
#    ips/cards, transacting rapidly, at odd hours, on cash-out-friendly
#    categories, in bursts.
# ---------------------------------------------------------------
for ring_idx in range(N_FRAUD_RINGS):
    ring_size = np.random.randint(*RING_SIZE_RANGE)
    ring_accounts = [new_id("acct") for _ in range(ring_size)]

    # small shared pool -> creates dense graph connectivity
    shared_devices = [new_id("dev") for _ in range(max(1, ring_size // 4))]
    shared_ips = [new_id("ip") for _ in range(max(1, ring_size // 3))]
    shared_cards = [new_id("card") for _ in range(max(1, ring_size // 2))]

    for account_id in ring_accounts:
        n_txns = np.random.randint(*FRAUD_TXNS_PER_ACCOUNT)
        for _ in range(n_txns):
            device_id = random.choice(shared_devices)
            ip_id = random.choice(shared_ips)
            card_id = random.choice(shared_cards)
            # fraud rings deliberately mimic normal behavior at the transaction
            # level (same hour/amount/velocity distributions as normal users) --
            # they only get caught by WHO they share infrastructure with, not
            # by looking individually suspicious. This is what makes the graph
            # signal necessary rather than just a bonus feature.
            hour = np.random.randint(0, 24)
            day = np.random.randint(1, 31)
            amount = round(np.random.lognormal(mean=3.8, sigma=1.1), 2)
            category = (np.random.choice(FRAUD_PREFERRED_CATEGORIES) if np.random.random() < 0.45
                        else np.random.choice(MERCHANT_CATEGORIES))
            time_since_last = np.random.randint(600, 3600 * 60)
            txn_count_24h = np.random.randint(1, 7)

            rows.append({
                "transaction_id": new_id("txn"),
                "account_id": account_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "card_id": card_id,
                "merchant_id": new_id("merch"),
                "amount": amount,
                "hour_of_day": int(hour),
                "day": day,
                "is_weekend": int(day % 7 in (0, 6)),
                "merchant_category": category,
                "time_since_last_txn_sec": time_since_last,
                "txn_count_last_24h": txn_count_24h,
                "is_fraud": 1
            })

# ---------------------------------------------------------------
# 3. REALISTIC NOISE: a handful of normal accounts legitimately share
#    a device/card (e.g. family members, shared work laptop). This
#    means "shared entity" alone isn't a perfect fraud tell -- the
#    model has to weigh graph signal together with behavior, not use
#    it as a lookup table. Ring sizes here are small (2, occasionally 3).
# ---------------------------------------------------------------
normal_account_ids = [r["account_id"] for r in rows if r["is_fraud"] == 0]
unique_normal_accounts = list(dict.fromkeys(normal_account_ids))
N_FAMILY_GROUPS = 60

for _ in range(N_FAMILY_GROUPS):
    group_size = np.random.choice([2, 2, 2, 3], p=[0.5, 0.2, 0.2, 0.1])
    members = random.sample(unique_normal_accounts, group_size)
    shared_device = new_id("dev")
    shared_card = new_id("card") if np.random.random() < 0.5 else None

    for account_id in members:
        n_txns = np.random.randint(1, 4)
        for _ in range(n_txns):
            rows.append({
                "transaction_id": new_id("txn"),
                "account_id": account_id,
                "device_id": shared_device,
                "ip_id": new_id("ip"),
                "card_id": shared_card if shared_card else new_id("card"),
                "merchant_id": new_id("merch"),
                "amount": round(np.random.lognormal(mean=3.7, sigma=1.1), 2),
                "hour_of_day": np.random.randint(0, 24),
                "day": np.random.randint(1, 31),
                "is_weekend": int(np.random.randint(1, 31) % 7 in (0, 6)),
                "merchant_category": np.random.choice(MERCHANT_CATEGORIES),
                "time_since_last_txn_sec": np.random.randint(600, 3600 * 72),
                "txn_count_last_24h": np.random.randint(1, 6),
                "is_fraud": 0
            })

df = pd.DataFrame(rows).sample(frac=1, random_state=RNG_SEED).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Fraud rate: {:.3%}".format(df["is_fraud"].mean()))
print(df["is_fraud"].value_counts())

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
df.to_csv(os.path.join(BASE_DIR, "data", "transactions.csv"), index=False)
print("\nSaved to data/transactions.csv")
