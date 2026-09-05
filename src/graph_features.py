"""
WebLock - Graph Construction + Feature Engineering
----------------------------------------------------
Builds a bipartite-derived graph connecting ACCOUNTS that share a
device, IP, or card, then computes graph-theoretic features per account:

  - degree: how many other accounts this one is directly linked to
  - component_size: size of the connected cluster this account belongs to
    (a large component = likely fraud ring)
  - pagerank: importance/centrality in the shared-entity network
  - clustering_coefficient: how tightly-knit the account's neighborhood is
  - n_shared_devices / n_shared_ips / n_shared_cards: raw sharing counts

These features get joined back onto the transaction-level table and fed
into the classifier alongside standard tabular features.
"""

import pandas as pd
import networkx as nx
import os
from itertools import combinations

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "transactions.csv"))

# On real-world data, some entity values are shared by thousands of accounts
# (e.g. a common card bin, or a generic device string) -- these are NOT fraud
# rings, they're just low-information generic values. Connecting every pair
# in a group that large would both blow up runtime (a group of 5,000 accounts
# is ~12.5 million pairs) and pollute the graph with meaningless edges. We
# cap group size: entities shared by more accounts than this are skipped
# entirely and treated as uninformative, same as a real production system
# would need to do.
MAX_ENTITY_GROUP_SIZE = 50

# ---------------------------------------------------------------
# Build account-level graph: edge between two accounts if they share
# a device, ip, or card at least once.
# ---------------------------------------------------------------
G = nx.Graph()
G.add_nodes_from(df["account_id"].unique())

shared_counts = {"device_id": 0, "ip_id": 0, "card_id": 0}
skipped_counts = {"device_id": 0, "ip_id": 0, "card_id": 0}

for entity_col in ["device_id", "ip_id", "card_id"]:
    groups = df.groupby(entity_col)["account_id"].unique()
    for accounts in groups:
        n = len(accounts)
        if n > 1:
            if n > MAX_ENTITY_GROUP_SIZE:
                skipped_counts[entity_col] += 1
                continue
            shared_counts[entity_col] += 1
            for a, b in combinations(sorted(set(accounts)), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b][entity_col] = G[a][b].get(entity_col, 0) + 1
                else:
                    G.add_edge(a, b, weight=1, **{entity_col: 1})

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Shared entities used by >1 account (kept): {shared_counts}")
print(f"Shared entities skipped (group size > {MAX_ENTITY_GROUP_SIZE}, treated as generic/uninformative): {skipped_counts}")

# ---------------------------------------------------------------
# Graph features per account
# ---------------------------------------------------------------
degree = dict(G.degree())
pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_edges() > 0 else {n: 0 for n in G.nodes()}
clustering = nx.clustering(G)

component_size = {}
for component in nx.connected_components(G):
    size = len(component)
    for node in component:
        component_size[node] = size

graph_feat_df = pd.DataFrame({
    "account_id": list(G.nodes()),
    "degree": [degree.get(n, 0) for n in G.nodes()],
    "pagerank": [pagerank.get(n, 0) for n in G.nodes()],
    "clustering_coefficient": [clustering.get(n, 0) for n in G.nodes()],
    "component_size": [component_size.get(n, 1) for n in G.nodes()],
})

print("\nGraph feature distribution:")
print(graph_feat_df.describe())

# join back to fraud label for a sanity check (mean fraud rate by component_size bucket)
acct_fraud = df.groupby("account_id")["is_fraud"].max().reset_index()
check = graph_feat_df.merge(acct_fraud, on="account_id")
print("\n=== Sanity check: fraud rate by component_size bucket ===")
check["component_bucket"] = pd.cut(check["component_size"], bins=[0,1,2,5,20], labels=["1 (isolated)","2","3-5","6+"])
print(check.groupby("component_bucket")["is_fraud"].mean())

# ---------------------------------------------------------------
# Merge graph features onto transaction-level data + save
# ---------------------------------------------------------------
full_df = df.merge(graph_feat_df, on="account_id", how="left")
full_df.to_csv(os.path.join(BASE_DIR, "data", "transactions_with_graph_features.csv"), index=False)
print("\nSaved: data/transactions_with_graph_features.csv")
print("Final shape:", full_df.shape)
