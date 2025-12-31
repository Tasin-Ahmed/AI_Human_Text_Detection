import networkx as nx
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import random
import time
import warnings
from transformers import BertTokenizer, BertModel  # Use specific PyTorch components

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "sagorsarker/bangla-bert-base"
OUTPUT_DIR = "bangla_outputs_AI"
OUTPUT_EXCEL = os.path.join(OUTPUT_DIR, "Jharna_Rahman_AI.xlsx")

# Thresholds for switching to approximate computations
MAX_NODES_EXACT = 500           # <= this: do exact heavy computations
BETWEENNESS_SAMPLE_K = 100      # 'k' parameter for approx betweenness
PATHS_SAMPLE_NODES = 100        # number of source nodes to sample for path-based metrics
RANDOM_SEED = 42

# create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------
# load model (PyTorch only)
# ---------------------------
print("Loading tokenizer & model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.\n")

# ---------------------------
# helper embedding function
# ---------------------------
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

# ---------------------------
# utilities for approximations
# ---------------------------
def sample_nodes_list(G, k):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    if n <= k:
        return nodes
    return random.sample(nodes, k)

def approx_average_shortest_path(H, sample_nodes):
    lengths_accum = []
    count = 0
    for s in sample_nodes:
        try:
            lens = nx.single_source_dijkstra_path_length(H, s, weight="distance")
            lengths_accum.extend(list(lens.values()))
            count += len(lens)
        except Exception:
            continue
    if count == 0:
        return -1
    return float(sum(lengths_accum) / count)

def approx_wiener_index(H, sample_nodes):
    wsum = 0.0
    pairs_count = 0
    for s in sample_nodes:
        try:
            lens = nx.single_source_dijkstra_path_length(H, s, weight="distance")
        except Exception:
            continue
        for t, d in lens.items():
            if s < t:
                wsum += d
                pairs_count += 1
    return float(wsum) if pairs_count > 0 else -1

def approx_eccentricity(H, sample_nodes):
    ecc = {}
    for s in sample_nodes:
        try:
            lens = nx.single_source_dijkstra_path_length(H, s, weight="distance")
            if len(lens) == 0:
                ecc[s] = -1
            else:
                ecc[s] = max(lens.values())
        except Exception:
            ecc[s] = -1
    return ecc

# ---------------------------
# Graph metrics computation
# ---------------------------
def compute_graph_metrics(G, H, name, max_nodes_exact=MAX_NODES_EXACT,
                         betweenness_sample_k=BETWEENNESS_SAMPLE_K,
                         paths_sample_nodes=PATHS_SAMPLE_NODES):
    metrics = {"Graph": name}

    # Basic properties
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    metrics["number_of_nodes"] = n_nodes
    metrics["number_of_edges"] = n_edges
    metrics["density"] = nx.density(G) if n_nodes > 1 else 0
    metrics["connected"] = nx.is_connected(G) if n_nodes > 0 else False
    metrics["bipartite"] = nx.is_bipartite(G) if n_nodes > 0 else False

    try:
        metrics["planar"] = nx.check_planarity(G)[0]
    except Exception:
        metrics["planar"] = False

    # Components
    components = list(nx.connected_components(G))
    metrics["number_connected_components"] = len(components)
    largest_cc = max(components, key=len) if components else set()
    metrics["size_largest_component"] = len(largest_cc) if largest_cc else 0

    # Prepare largest connected component for path-based metrics
    if metrics["connected"]:
        H_lcc = H
    else:
        H_lcc = H.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    n_lcc = H_lcc.number_of_nodes()
    use_exact_paths = (n_lcc <= max_nodes_exact) and (n_lcc > 0)

    # Path-based metrics
    if n_lcc == 0:
        metrics["diameter"] = -1
        metrics["radius"] = -1
        metrics["avg_shortest_path_length"] = -1
        metrics["wiener_index"] = -1
    else:
        if use_exact_paths:
            try:
                metrics["diameter"] = nx.diameter(H_lcc, weight='distance')
                metrics["radius"] = nx.radius(H_lcc, weight='distance')
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(H_lcc, weight='distance')
                # Wiener index calculation
                wiener = 0
                paths = dict(nx.all_pairs_dijkstra_path_length(H_lcc, weight='distance'))
                nodes_list = list(H_lcc.nodes())
                for i in range(len(nodes_list)):
                    for j in range(i+1, len(nodes_list)):
                        u = nodes_list[i]
                        v = nodes_list[j]
                        wiener += paths[u][v]
                metrics["wiener_index"] = wiener
            except Exception:
                metrics["diameter"] = -1
                metrics["radius"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["wiener_index"] = -1
        else:
            sample_nodes = sample_nodes_list(H_lcc, min(paths_sample_nodes, n_lcc))
            ecc = approx_eccentricity(H_lcc, sample_nodes)
            if ecc:
                metrics["diameter"] = max(ecc.values()) if ecc else -1
                metrics["radius"] = min(ecc.values()) if ecc else -1
            else:
                metrics["diameter"] = -1
                metrics["radius"] = -1
            metrics["avg_shortest_path_length"] = approx_average_shortest_path(H_lcc, sample_nodes)
            metrics["wiener_index"] = approx_wiener_index(H_lcc, sample_nodes)

    # Girth
    try:
        cycles = nx.cycle_basis(G)
        if cycles:
            metrics["girth"] = min(len(cycle) for cycle in cycles)
        else:
            metrics["girth"] = -1
    except Exception:
        metrics["girth"] = -1

    # Centrality metrics
    try:
        deg_cent = nx.degree_centrality(G)
        metrics["avg_degree_centrality"] = np.mean(list(deg_cent.values())) if deg_cent else -1
    except Exception:
        metrics["avg_degree_centrality"] = -1

    try:
        if n_nodes <= max_nodes_exact:
            betweenness = nx.betweenness_centrality(G)
        else:
            betweenness = nx.betweenness_centrality(G, k=betweenness_sample_k)
        metrics["avg_betweenness_centrality"] = np.mean(list(betweenness.values())) if betweenness else -1
    except Exception:
        metrics["avg_betweenness_centrality"] = -1

    try:
        if n_nodes <= max_nodes_exact:
            closeness = nx.closeness_centrality(G)
            metrics["avg_closeness_centrality"] = np.mean(list(closeness.values())) if closeness else -1
        else:
            metrics["avg_closeness_centrality"] = -1
    except Exception:
        metrics["avg_closeness_centrality"] = -1

    # Clustering and transitivity
    try:
        metrics["transitivity"] = nx.transitivity(G)
    except Exception:
        metrics["transitivity"] = -1

    try:
        clustering = nx.clustering(G)
        metrics["avg_clustering"] = np.mean(list(clustering.values())) if clustering else -1
    except Exception:
        metrics["avg_clustering"] = -1

    # Degeneracy
    try:
        if n_nodes <= max_nodes_exact:
            core_num = nx.core_number(G)
            metrics["degeneracy"] = max(core_num.values()) if core_num else -1
        else:
            metrics["degeneracy"] = -1
    except Exception:
        metrics["degeneracy"] = -1

    # Clique number
    try:
        if n_nodes <= max_nodes_exact:
            metrics["clique_number"] = nx.graph_clique_number(G)
        else:
            metrics["clique_number"] = -1
    except Exception:
        metrics["clique_number"] = -1

    # Spectral properties
    try:
        if n_nodes <= max_nodes_exact and n_nodes > 0:
            A = nx.to_numpy_array(G)
            eigenvalues = np.linalg.eigvals(A).real
            metrics["graph_energy"] = np.sum(np.abs(eigenvalues))
            sorted_eigen = np.sort(eigenvalues)
            metrics["top_eigenvalue"] = sorted_eigen[-1] if len(eigenvalues) > 0 else -1
            metrics["second_eigenvalue"] = sorted_eigen[-2] if len(eigenvalues) > 1 else -1
        else:
            metrics["graph_energy"] = -1
            metrics["top_eigenvalue"] = -1
            metrics["second_eigenvalue"] = -1
    except Exception:
        metrics["graph_energy"] = -1
        metrics["top_eigenvalue"] = -1
        metrics["second_eigenvalue"] = -1

    # Max flow
    try:
        if n_nodes >= 2:
            nodes_list = list(G.nodes())
            flow_value = nx.maximum_flow_value(G, nodes_list[0], nodes_list[1], capacity='weight')
            metrics["max_flow"] = flow_value
        else:
            metrics["max_flow"] = -1
    except Exception:
        metrics["max_flow"] = -1

    # Matching number
    try:
        if n_nodes > 0:
            matching = nx.max_weight_matching(G)
            metrics["matching_number"] = len(matching)  # Number of edges in matching
        else:
            metrics["matching_number"] = -1
    except Exception:
        metrics["matching_number"] = -1

    # Algebraic connectivity
    try:
        if n_nodes > 0 and nx.is_connected(G):
            metrics["algebraic_connectivity"] = nx.algebraic_connectivity(nx.to_numpy_array(G))
        else:
            metrics["algebraic_connectivity"] = -1
    except Exception:
        metrics["algebraic_connectivity"] = -1

    return metrics

# ---------------------------
# SAMPLE PARAGRAPHS
# ---------------------------
paragraphs = [
 """
আমার নাম আরিয়ান। আমি ক্লাস টেনের ছাত্র। আমি ছোটবেলা থেকেই সূর্যাস্তের সময়ের ছায়া নিয়ে কৌতূহলী—দেখেছো তো, বিকেলের আলোয় ছায়া লম্বা হয়ে যায়? কিন্তু সেদিন আমি এমন কিছু দেখেছিলাম যা আমার জীবন বদলে দিয়েছে।
ঘটনাটা ঘটেছিল গত শনিবার। বিকেল পাঁচটা নাগাদ আমি ছাদে বসে ছবি তুলছিলাম। হঠাৎ লক্ষ্য করলাম, নিচে রাস্তার একপাশে দাঁড়ানো ছায়াগুলো অদ্ভুতভাবে নড়ছে—কিন্তু যাদের ছায়া, তারা মোটেও নড়ছে না!
প্রথমে ভেবেছিলাম হয়তো আলো বা বাতাসের কোনো খেলা। কিন্তু ভালো করে তাকিয়ে দেখি, একটি ছায়া ধীরে ধীরে তার আসল মানুষের থেকে আলাদা হয়ে যাচ্ছে। তারপর সেটা রাস্তার অন্যপাশে হেঁটে গেল।
আমার বুক ধপধপ করতে লাগলো। ক্যামেরা দিয়ে জুম করে ছবি তুলতে যাচ্ছি, ঠিক তখনই ছায়াটা মাথা ঘুরিয়ে সরাসরি আমার দিকে তাকালো! কেমন যেন শীতল অনুভূতি হলো সারা শরীরে—যেন ওটা জানে আমি ওকে দেখছি।
আমি তাড়াতাড়ি নিচে নেমে গেলাম। রাস্তায় গিয়ে দেখি স্বাভাবিক আলো-ছায়া, কিছুই অদ্ভুত না। কিন্তু মনের ভেতরে অস্বস্তি রয়ে গেল।
পরের দিন স্কুলে গিয়ে এই কথা আমার বন্ধু রিয়াদের কাছে বললাম। ও হাসতে হাসতে বললো, “তুমি নিশ্চয়ই বেশি সায়েন্স ফিকশন পড়ছো। ছায়া আবার হাঁটে নাকি!”
কিন্তু দুই দিন পর রিয়াদই আমাকে ফোন করে হাপাতে হাপাতে বললো, “আরিয়ান… আমি ওদের দেখেছি!”
আমি ছুটে গেলাম ওর বাসায়। রিয়াদ জানাল, গতরাতে সে দেখেছে তার ঘরের দেয়ালে নিজের ছায়া নেই—কিন্তু জানালার বাইরে দাঁড়িয়ে আছে, তাকে দেখছে। আর সকালে ঘুম থেকে উঠে দেখে তার শরীরে কয়েকটা কালো দাগ, যেন কেউ ছুঁয়ে দিয়েছে।
আমরা দুজনেই ভয় পেয়ে গেলাম। ঠিক করলাম, রাতে বাইরে গিয়ে ওদের খুঁজব।
সেই রাতেই আমরা স্কুল মাঠে গেলাম। মাঠে চারপাশে লাইটপোস্ট আছে, তাই অনেক ছায়া পড়ছে। কিছুক্ষণ পরই দেখলাম—দূরে একজন মানুষের ছায়া তার মালিকের থেকে আলাদা হয়ে যাচ্ছে। তারপর একে একে আরও কয়েকটা ছায়া আলাদা হলো, আর তারা সবাই একসাথে মাঠের মাঝখানে জড়ো হতে লাগলো।
আমরা লুকিয়ে দেখছিলাম। হঠাৎ ছায়াগুলো মিলেমিশে এক বিশাল কালো অবয়বে রূপ নিলো—মানুষের মতো, কিন্তু চোখ নেই, মুখ নেই।
"""
]

# storage
graph_features_list = []

# ---------------------------
# MAIN LOOP
# ---------------------------
t_start_all = time.time()
for idx, paragraph in enumerate(paragraphs, 1):
    t0 = time.time()
    print(f"\n--- Paragraph {idx} ---")
    cleaned = re.sub(r"[।,?!.;:“”‘’\"()\[\]—–\-…]", "", paragraph)
    words = cleaned.split()
    if not words:
        print(f"Paragraph {idx} is empty — skipping.")
        continue

    print(f"Tokens: {len(words)} — computing embeddings...")
    word_embeddings = []
    for w in words:
        try:
            emb = get_word_embedding(w)
        except Exception as e:
            print(f"Embedding error for word {w!r}: {e}. Using zeros.")
            emb = np.zeros((model.config.hidden_size,))
        word_embeddings.append(emb)

    sim_mat = cosine_similarity(word_embeddings)

    G = nx.Graph()
    for widx, word in enumerate(words):
        G.add_node(widx, word=word)

    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w = float(sim_mat[i, j])
            if w > 0.5:
                G.add_edge(i, j, weight=round(w, 4))

    H = G.copy()
    for u, v, data in H.edges(data=True):
        wt = data.get("weight", 1e-6)
        data["distance"] = 1.0 / (wt if wt != 0 else 1e-6)

    # Compute comprehensive graph metrics
    graph_data = compute_graph_metrics(G, H, idx)
    graph_features_list.append(graph_data)

    t1 = time.time()
    print(f"Paragraph {idx} done in {t1 - t0:.1f}s.")

# ----------------- CREATE FINAL TABLE -----------------
df_graphs = pd.DataFrame(graph_features_list)
avg_row = df_graphs.mean(numeric_only=True)
avg_row["Graph"] = "Average"
df_graphs = pd.concat([df_graphs, pd.DataFrame([avg_row])], ignore_index=True)

# Column order
cols = [
    "Graph", "number_of_nodes", "number_of_edges", "density", "connected",
    "bipartite", "planar", "number_connected_components", "size_largest_component",
    "diameter", "radius", "avg_shortest_path_length", "wiener_index", "girth",
    "avg_degree_centrality", "avg_betweenness_centrality", "avg_closeness_centrality",
    "transitivity", "avg_clustering", "degeneracy", "clique_number", "graph_energy",
    "top_eigenvalue", "second_eigenvalue", "max_flow", "matching_number", "algebraic_connectivity"
]
df_graphs = df_graphs[cols]

# ----------------- WRITE TO EXCEL -----------------
print("\nWriting to Excel...")
df_graphs.to_excel(OUTPUT_EXCEL, index=False)

print(f"✅ Graph properties saved to {OUTPUT_EXCEL}")
print(f"Total time: {time.time() - t_start_all:.1f}s")