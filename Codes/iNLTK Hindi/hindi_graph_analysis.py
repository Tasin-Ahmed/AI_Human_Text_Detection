"""
Hindi Graph Analysis with iNLTK
Python 3.8/3.9 Compatible Version
"""

import networkx as nx
import pandas as pd
import numpy as np
import re
import os
import random
import time
import warnings
import unicodedata
from collections import defaultdict, Counter
import itertools
import math

# Import iNLTK
try:
    from inltk.inltk import tokenize, setup
    print("✅ iNLTK imported successfully!")
except ImportError:
    print("❌ ERROR: iNLTK not installed!")
    print("Please run: pip install inltk==0.7.6")
    exit(1)

# Import indic-nlp for better preprocessing
try:
    from indicnlp.tokenize import indic_tokenize
    from indicnlp.normalize import indic_normalize
    print("✅ Indic-NLP imported successfully!")
    HAS_INDICNLP = True
except ImportError:
    print("⚠️  Warning: indic-nlp-library not found. Using basic preprocessing.")
    HAS_INDICNLP = False

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG
# ---------------------------
OUTPUT_DIR = "hindi_outputs_PMI"
INPUT_HUMAN = "Hindi_Human.csv"
INPUT_AI = "Hindi_AI.csv"

# Graph computation thresholds
MAX_NODES_EXACT = 500
BETWEENNESS_SAMPLE_K = 100
PATHS_SAMPLE_NODES = 100
RANDOM_SEED = 42

# PMI Parameters
WINDOW_SIZE = 4
MIN_PMI = 0.0

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("\n" + "="*70)
print("Hindi NLP Graph Analysis with iNLTK")
print("="*70 + "\n")

# ---------------------------
# HINDI STOPWORDS
# ---------------------------
HINDI_STOPWORDS = {
    'का', 'के', 'की', 'को', 'से', 'में', 'पर', 'है', 'हैं', 'था', 'थे', 'थी',
    'एक', 'यह', 'वह', 'जो', 'कि', 'और', 'या', 'तथा', 'इस', 'उस', 'ने',
    'कर', 'कुछ', 'सकता', 'सकते', 'होता', 'होती', 'हो', 'गया', 'गयी', 'गए',
    'हूँ', 'हूं', 'हैं', 'हो', 'था', 'थी', 'थे', 'बहुत', 'जब', 'तब', 'क्या',
    'कौन', 'कहाँ', 'कैसे', 'किस', 'किसी', 'अपना', 'अपने', 'अपनी', 'सब',
    'यदि', 'जैसे', 'वाला', 'वाले', 'वाली', 'दो', 'तीन', 'चार', 'पांच',
    'इसलिए', 'क्योंकि', 'लेकिन', 'परन्तु', 'अगर', 'मगर', 'तो', 'ही', 'भी',
    'न', 'नहीं', 'कभी', 'फिर', 'अब', 'जा', 'रहा', 'रही', 'रहे', 'गा', 'गी', 'गे',
    'हुआ', 'हुई', 'हुए', 'करना', 'किया', 'किये', 'करते', 'करती', 'होने', 'वाले',
    'साथ', 'द्वारा', 'लिए', 'बाद', 'पहले', 'दौरान', 'तक', 'बीच', 'ऊपर', 'नीचे'
}

# ---------------------------
# TEXT CLEANING
# ---------------------------
def clean_hindi_text(text):
    """Advanced cleaning for Hindi text"""
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove punctuation
    text = re.sub(r'[।,?!.;:"''""\(\)\[\]—–\-…]', '', text)
    
    # Keep only Devanagari characters and spaces
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ---------------------------
# SIMPLE HINDI LEMMATIZER
# ---------------------------
def simple_hindi_lemmatize(word):
    """Rule-based Hindi lemmatization"""
    if len(word) <= 2:
        return word
    
    # Common Hindi suffixes (ordered by length - remove longest first)
    suffixes = [
        'ों', 'ओं', 'ाओं', 'ियों', 'ुओं',  # Plural markers
        'ें', 'ीं', 'ाएं', 'ाईं',           # Feminine plural
        'ाया', 'ाये', 'ाई', 'ाए', 'ाओ',    # Verb forms
        'ता', 'ती', 'ते', 'ना',              # Verb endings
        'ने', 'नी', 'या', 'यी', 'ये',        # More verb forms
        'ान', 'ीय', 'िक',                    # Adjective/noun suffixes
        'ें', 'ों'                            # Case markers
    ]
    
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[:-len(suffix)]
    
    return word

# ---------------------------
# INLTK PREPROCESSING
# ---------------------------
def preprocess_inltk(text):
    """
    Process Hindi text using iNLTK
    Returns: List of processed tokens
    """
    try:
        # Clean text
        cleaned = clean_hindi_text(text)
        
        if not cleaned.strip():
            return []
        
        # Normalize using Indic-NLP if available
        if HAS_INDICNLP:
            try:
                normalizer = indic_normalize.IndicNormalizerFactory().get_normalizer("hi")
                cleaned = normalizer.normalize(cleaned)
            except:
                pass
        
        # Tokenize using iNLTK
        tokens = tokenize(cleaned, 'hi')
        
        if not tokens:
            return []
        
        # Remove stopwords and short tokens
        filtered_tokens = [
            token for token in tokens 
            if token not in HINDI_STOPWORDS and len(token) > 1
        ]
        
        # Apply lemmatization
        lemmatized = [simple_hindi_lemmatize(token) for token in filtered_tokens]
        
        # Final filtering - remove any tokens that became too short or are stopwords
        final_tokens = [
            token for token in lemmatized 
            if len(token) > 1 and token not in HINDI_STOPWORDS
        ]
        
        return final_tokens
        
    except Exception as e:
        print(f"Error in iNLTK preprocessing: {e}")
        return []

# ---------------------------
# READ INPUT FILE (CSV/EXCEL)
# ---------------------------
def read_file_paragraphs(filepath):
    """Read paragraphs from single-column CSV or Excel file"""
    try:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return []

        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            try:
                # Try reading with default encoding
                df = pd.read_csv(filepath)
            except UnicodeDecodeError:
                # Fallback to latin-1 if utf-8 fails
                df = pd.read_csv(filepath, encoding='latin-1')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            print(f"❌ Unsupported file extension: {ext}")
            return []

        # Assume the text is in the first column
        paragraphs = df.iloc[:, 0].dropna().astype(str)
        paragraphs = [p for p in paragraphs if p.strip()]
        return paragraphs

    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return []

# ---------------------------
# PMI CO-OCCURRENCE
# ---------------------------
def calculate_pmi_edges(tokens, window_size=WINDOW_SIZE, min_pmi=MIN_PMI):
    """Calculate PMI-based edges from tokens"""
    if len(tokens) < 2:
        return {}

    word_freq = Counter()
    pair_freq = Counter()
    total_windows = 0

    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        total_windows += 1

        for word in window:
            word_freq[word] += 1

        for w1, w2 in itertools.combinations(window, 2):
            pair = tuple(sorted((w1, w2)))
            pair_freq[pair] += 1

    edges = {}
    for (w1, w2), c in pair_freq.items():
        if total_windows == 0:
            continue

        p_xy = c / total_windows
        p_x = word_freq[w1] / (total_windows * window_size)
        p_y = word_freq[w2] / (total_windows * window_size)

        if p_x > 0 and p_y > 0:
            pmi = math.log(p_xy / (p_x * p_y))
            if pmi > min_pmi:
                edges[(w1, w2)] = pmi

    return edges

def create_pmi_graph(tokens):
    """Create NetworkX graph from tokens using PMI"""
    edges = calculate_pmi_edges(tokens)
    G = nx.Graph()

    for token in set(tokens):
        G.add_node(token)

    for (w1, w2), pmi_score in edges.items():
        G.add_edge(w1, w2, weight=pmi_score)

    return G

# ---------------------------
# APPROXIMATION UTILITIES
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
# GRAPH METRICS COMPUTATION
# ---------------------------
def compute_graph_metrics(G, H, name):
    """Compute comprehensive graph metrics"""
    metrics = {"Graph": name}

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

    components = list(nx.connected_components(G))
    metrics["number_connected_components"] = len(components)
    largest_cc = max(components, key=len) if components else set()
    metrics["size_largest_component"] = len(largest_cc) if largest_cc else 0

    if metrics["connected"]:
        H_lcc = H
    else:
        H_lcc = H.subgraph(largest_cc).copy() if largest_cc else nx.Graph()

    n_lcc = H_lcc.number_of_nodes()
    use_exact = (n_lcc <= MAX_NODES_EXACT) and (n_lcc > 0)

    if n_lcc == 0:
        metrics["diameter"] = -1
        metrics["radius"] = -1
        metrics["avg_shortest_path_length"] = -1
        metrics["wiener_index"] = -1
    else:
        if use_exact:
            try:
                metrics["diameter"] = nx.diameter(H_lcc, weight='distance')
                metrics["radius"] = nx.radius(H_lcc, weight='distance')
                metrics["avg_shortest_path_length"] = nx.average_shortest_path_length(H_lcc, weight='distance')
                
                wiener = 0
                paths = dict(nx.all_pairs_dijkstra_path_length(H_lcc, weight='distance'))
                nodes_list = list(H_lcc.nodes())
                for i in range(len(nodes_list)):
                    for j in range(i+1, len(nodes_list)):
                        u, v = nodes_list[i], nodes_list[j]
                        wiener += paths[u][v]
                metrics["wiener_index"] = wiener
            except Exception:
                metrics["diameter"] = -1
                metrics["radius"] = -1
                metrics["avg_shortest_path_length"] = -1
                metrics["wiener_index"] = -1
        else:
            sample_nodes = sample_nodes_list(H_lcc, min(PATHS_SAMPLE_NODES, n_lcc))
            ecc = approx_eccentricity(H_lcc, sample_nodes)
            if ecc:
                metrics["diameter"] = max(ecc.values()) if ecc else -1
                metrics["radius"] = min(ecc.values()) if ecc else -1
            else:
                metrics["diameter"] = -1
                metrics["radius"] = -1
            metrics["avg_shortest_path_length"] = approx_average_shortest_path(H_lcc, sample_nodes)
            metrics["wiener_index"] = approx_wiener_index(H_lcc, sample_nodes)

    try:
        cycles = nx.cycle_basis(G)
        metrics["girth"] = min(len(cycle) for cycle in cycles) if cycles else -1
    except Exception:
        metrics["girth"] = -1

    try:
        deg_cent = nx.degree_centrality(G)
        metrics["avg_degree_centrality"] = np.mean(list(deg_cent.values())) if deg_cent else -1
    except Exception:
        metrics["avg_degree_centrality"] = -1

    try:
        if n_nodes <= MAX_NODES_EXACT:
            betweenness = nx.betweenness_centrality(G)
        else:
            betweenness = nx.betweenness_centrality(G, k=BETWEENNESS_SAMPLE_K)
        metrics["avg_betweenness_centrality"] = np.mean(list(betweenness.values())) if betweenness else -1
    except Exception:
        metrics["avg_betweenness_centrality"] = -1

    try:
        if n_nodes <= MAX_NODES_EXACT:
            closeness = nx.closeness_centrality(G)
            metrics["avg_closeness_centrality"] = np.mean(list(closeness.values())) if closeness else -1
        else:
            metrics["avg_closeness_centrality"] = -1
    except Exception:
        metrics["avg_closeness_centrality"] = -1

    try:
        metrics["transitivity"] = nx.transitivity(G)
    except Exception:
        metrics["transitivity"] = -1

    try:
        clustering = nx.clustering(G)
        metrics["avg_clustering"] = np.mean(list(clustering.values())) if clustering else -1
    except Exception:
        metrics["avg_clustering"] = -1

    try:
        if n_nodes <= MAX_NODES_EXACT:
            core_num = nx.core_number(G)
            metrics["degeneracy"] = max(core_num.values()) if core_num else -1
        else:
            metrics["degeneracy"] = -1
    except Exception:
        metrics["degeneracy"] = -1

    try:
        if n_nodes <= MAX_NODES_EXACT:
            metrics["clique_number"] = nx.graph_clique_number(G)
        else:
            metrics["clique_number"] = -1
    except Exception:
        metrics["clique_number"] = -1

    try:
        if n_nodes <= MAX_NODES_EXACT and n_nodes > 0:
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

    try:
        if n_nodes >= 2:
            nodes_list = list(G.nodes())
            flow_value = nx.maximum_flow_value(G, nodes_list[0], nodes_list[1], capacity='weight')
            metrics["max_flow"] = flow_value
        else:
            metrics["max_flow"] = -1
    except Exception:
        metrics["max_flow"] = -1

    try:
        if n_nodes > 0:
            matching = nx.max_weight_matching(G)
            metrics["matching_number"] = len(matching)
        else:
            metrics["matching_number"] = -1
    except Exception:
        metrics["matching_number"] = -1

    try:
        if n_nodes > 0 and nx.is_connected(G):
            metrics["algebraic_connectivity"] = nx.algebraic_connectivity(G)
        else:
            metrics["algebraic_connectivity"] = -1
    except Exception:
        metrics["algebraic_connectivity"] = -1

    return metrics

# ---------------------------
# MAIN PROCESSING
# ---------------------------
def process_file(filepath, source_name):
    """Process one input file"""
    print(f"\n{'='*70}")
    print(f"Processing: {source_name}")
    print(f"{'='*70}")

    paragraphs = read_file_paragraphs(filepath)

    if not paragraphs:
        print(f"❌ No paragraphs found in {filepath}")
        return None

    print(f"📄 Found {len(paragraphs)} paragraphs\n")

    graph_features_list = []
    t_start = time.time()

    for idx, paragraph in enumerate(paragraphs, 1):
        t0 = time.time()
        print(f"[{idx}/{len(paragraphs)}] ", end="", flush=True)

        tokens = preprocess_inltk(paragraph)

        if not tokens:
            print("⚠️  Empty after preprocessing - skipping")
            continue

        print(f"{len(tokens)} tokens → ", end="", flush=True)

        G = create_pmi_graph(tokens)
        print(f"Graph({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) ", end="", flush=True)

        H = G.copy()
        for u, v, data in H.edges(data=True):
            wt = data.get("weight", 1e-6)
            data["distance"] = 1.0 / (wt if wt > 1e-6 else 1e-6)

        metrics = compute_graph_metrics(G, H, idx)
        graph_features_list.append(metrics)

        print(f"✓ ({time.time() - t0:.1f}s)")

    if not graph_features_list:
        print(f"\n❌ No valid graphs generated for {source_name}")
        return None

    # Create DataFrame
    df = pd.DataFrame(graph_features_list)

    # Add average row
    avg_row = df.mean(numeric_only=True)
    avg_row["Graph"] = "Average"
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Column order
    cols = [
        "Graph", "number_of_nodes", "number_of_edges", "density", "connected",
        "bipartite", "planar", "number_connected_components", "size_largest_component",
        "diameter", "radius", "avg_shortest_path_length", "wiener_index", "girth",
        "avg_degree_centrality", "avg_betweenness_centrality", "avg_closeness_centrality",
        "transitivity", "avg_clustering", "degeneracy", "clique_number", "graph_energy",
        "top_eigenvalue", "second_eigenvalue", "max_flow", "matching_number", "algebraic_connectivity"
    ]
    df = df[cols]

    # Save to Excel
    output_filename = f"{source_name}_inltk_PMI.xlsx"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.to_excel(output_path, index=False)

    print(f"\n✅ Saved: {output_path}")
    print(f"⏱️  Time: {time.time() - t_start:.1f}s")
    print(f"📊 Processed: {len(graph_features_list)} paragraphs")

    return df

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("Initializing iNLTK for Hindi...")
    try:
        setup('hi')
        print("✅ iNLTK Hindi model loaded!\n")
    except Exception as e:
        print(f"⚠️  iNLTK setup warning: {e}")
        print("Continuing anyway...\n")
    
    total_start = time.time()
    results = []

    # Process both files
    for filepath, source_name in [(INPUT_HUMAN, "human"), (INPUT_AI, "AI")]:
        if not os.path.exists(filepath):
            print(f"⚠️  {filepath} not found - skipping {source_name}")
            continue

        df = process_file(filepath, source_name)
        if df is not None:
            results.append({
                'Source': source_name,
                'Paragraphs': len(df) - 1,
                'Avg Nodes': f"{df['number_of_nodes'].iloc[-1]:.2f}",
                'Avg Edges': f"{df['number_of_edges'].iloc[-1]:.2f}"
            })

    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
    else:
        print("❌ No results generated")

    print(f"\n⏱️  Total time: {time.time() - total_start:.1f}s")
    print(f"📁 Output: {OUTPUT_DIR}/")
    print("\n🎉 Done!")