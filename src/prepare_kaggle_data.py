"""
WebLock - Kaggle Dataset Adapter (IEEE-CIS Fraud Detection)
----------------------------------------------------------------
Converts the real IEEE-CIS Fraud Detection dataset (train_transaction.csv
+ train_identity.csv from kaggle.com/c/ieee-fraud-detection) into the
same schema our pipeline expects, so graph_features.py / train_models.py /
explainability.py run completely unchanged on real data.

WHY THIS FILE EXISTS: IEEE-CIS doesn't give you a clean account_id,
device_id, ip_id like a toy dataset would -- real fraud data never does.
We have to construct sensible entity proxies from the columns available.
These choices are documented below so you can explain and defend them.

--------------------------------------------------------------------
ENTITY PROXY DECISIONS (be ready to explain these in an interview):
--------------------------------------------------------------------
- account_id  -> combination of (card1, card2, addr1). None of these alone
  identifies a person, but together they're a commonly used community
  heuristic (a "pseudo user ID") for grouping transactions that likely
  belong to the same real-world payer, since IEEE-CIS deliberately
  anonymizes and doesn't provide a direct user/account key.
- card_id     -> card1 (a high-cardinality anonymized numeric field Kaggle
  describes as core payment card info -- closest available proxy to a
  card identifier).
- device_id   -> DeviceInfo from the identity table (device/browser string).
  Only ~24% of transactions have identity data attached; rows without it
  get a unique placeholder so they don't get incorrectly clustered together.
- ip_id       -> P_emaildomain (purchaser's email domain). Not a literal IP,
  but shared email domains across accounts is itself a legitimate,
  real-world fraud-ring signal (e.g. disposable-email patterns), so it's
  used here as the second relational entity alongside device.
- merchant_category -> ProductCD (the dataset's product/category code).

These are reasonable, defensible modeling choices for a portfolio project,
not ground truth -- say so plainly if asked.
--------------------------------------------------------------------

Run this BEFORE graph_features.py, in place of generate_data.py:
    python src/prepare_kaggle_data.py
    python src/graph_features.py
    python src/train_models.py
    python src/explainability.py
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_DIR = os.path.join(BASE_DIR, "data", "kaggle")
TXN_PATH = os.path.join(KAGGLE_DIR, "train_transaction.csv")
ID_PATH = os.path.join(KAGGLE_DIR, "train_identity.csv")

# ---------------------------------------------------------------
# Sanity check the files exist before doing anything expensive
# ---------------------------------------------------------------
missing = [p for p in [TXN_PATH, ID_PATH] if not os.path.exists(p)]
if missing:
    print("ERROR: Missing Kaggle data file(s):")
    for m in missing:
        print(f"  - {m}")
    print("\nDownload train_transaction.csv and train_identity.csv from")
    print("kaggle.com/c/ieee-fraud-detection and place them in data/kaggle/")
    sys.exit(1)

# ---------------------------------------------------------------
# Load only the columns we actually need (the full file has 390+
# anonymized V-columns we don't use -- skipping them saves a lot
# of memory and load time)
# ---------------------------------------------------------------
print("Loading train_transaction.csv (this can take a minute)...")
txn_cols = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "addr1", "P_emaildomain"
]
txn = pd.read_csv(TXN_PATH, usecols=txn_cols)

print("Loading train_identity.csv...")
id_cols = ["TransactionID", "DeviceInfo"]
identity = pd.read_csv(ID_PATH, usecols=id_cols)

print(f"Transactions: {len(txn):,}  |  Identity records: {len(identity):,}")

# ---------------------------------------------------------------
# Merge (left join -- most transactions have NO identity record,
# that's expected and realistic)
# ---------------------------------------------------------------
df = txn.merge(identity, on="TransactionID", how="left")

# ---------------------------------------------------------------
# Build entity proxy columns
# ---------------------------------------------------------------
# account_id: combination of card1/card2/addr1
df["card1"] = df["card1"].fillna(-1).astype(int).astype(str)
df["card2"] = df["card2"].fillna(-1).astype(int).astype(str)
df["addr1"] = df["addr1"].fillna(-1).astype(int).astype(str)
df["account_id"] = "acct_" + df["card1"] + "_" + df["card2"] + "_" + df["addr1"]

# card_id: card1 alone
df["card_id"] = "card_" + df["card1"]

# device_id: DeviceInfo, with unique placeholders for missing values so
# they don't get falsely clustered together as if they were "the same device"
df["device_id"] = df["DeviceInfo"]
missing_device_mask = df["device_id"].isna()
df.loc[missing_device_mask, "device_id"] = (
    "unk_dev_" + df.loc[missing_device_mask, "TransactionID"].astype(str)
)

# ip_id proxy: purchaser email domain, same unique-placeholder logic
df["ip_id"] = df["P_emaildomain"]
missing_ip_mask = df["ip_id"].isna()
df.loc[missing_ip_mask, "ip_id"] = (
    "unk_ip_" + df.loc[missing_ip_mask, "TransactionID"].astype(str)
)

# merchant_category proxy
df["merchant_category"] = df["ProductCD"].fillna("unknown")

# amount, fraud label, transaction id
df["amount"] = df["TransactionAmt"]
df["is_fraud"] = df["isFraud"]
df["transaction_id"] = "txn_" + df["TransactionID"].astype(str)
df["merchant_id"] = df["transaction_id"]  # not used downstream, kept for schema compatibility

# ---------------------------------------------------------------
# Derive time features from TransactionDT (seconds elapsed from an
# arbitrary reference point -- NOT a real calendar timestamp, so
# hour/day/weekend are approximate, not literal calendar dates)
# ---------------------------------------------------------------
df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
df["day"] = (df["TransactionDT"] // 86400) % 30 + 1
df["is_weekend"] = (((df["TransactionDT"] // 86400) % 7).isin([5, 6])).astype(int)

# ---------------------------------------------------------------
# Velocity features: time since this account's last transaction,
# and how many transactions this account made in the prior 24h.
# Computed per account_id, sorted by time.
# ---------------------------------------------------------------
print("Computing per-account velocity features (this is the slow part)...")
df = df.sort_values(["account_id", "TransactionDT"]).reset_index(drop=True)

df["time_since_last_txn_sec"] = (
    df.groupby("account_id")["TransactionDT"].diff()
)
# first transaction for an account has no prior txn -- use a large sentinel value
df["time_since_last_txn_sec"] = df["time_since_last_txn_sec"].fillna(999_999).astype(int)


def rolling_24h_count(group):
    times = group["TransactionDT"].values
    counts = np.zeros(len(times), dtype=int)
    for i, t in enumerate(times):
        # count prior transactions (strictly before this one) within 24h
        window_start = t - 86400
        counts[i] = np.sum((times[:i] >= window_start))
    return counts


df["txn_count_last_24h"] = 0
counts_list = []
for _, group in df.groupby("account_id", sort=False):
    counts_list.append(pd.Series(rolling_24h_count(group), index=group.index))
df["txn_count_last_24h"] = pd.concat(counts_list).sort_index()

# ---------------------------------------------------------------
# Final schema, matching what graph_features.py / train_models.py expect
# ---------------------------------------------------------------
final_cols = [
    "transaction_id", "account_id", "device_id", "ip_id", "card_id", "merchant_id",
    "amount", "hour_of_day", "day", "is_weekend", "merchant_category",
    "time_since_last_txn_sec", "txn_count_last_24h", "is_fraud"
]
final_df = df[final_cols]

print(f"\nFinal dataset shape: {final_df.shape}")
print(f"Fraud rate: {final_df['is_fraud'].mean():.3%}")
print(final_df["is_fraud"].value_counts())

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
out_path = os.path.join(BASE_DIR, "data", "transactions.csv")
final_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
print("\nNext: run src/graph_features.py")
