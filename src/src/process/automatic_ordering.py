import networkx as nx
import cv2
from src.process import feature_extraction
from src.utils import image_utils
import numpy as np
import itertools

def hamiltonian_path_brute_force(G: nx.Graph) -> list[int]:
    n = G.number_of_nodes()
    best_path: list[int] | None = None
    
    min_cost = float('inf')  # safer than a hard-coded large number

    # Try all permutations of nodes
    for perm in itertools.permutations(G.nodes):
        path = list(perm)
        valid = True
        cost = 0

        # Check if consecutive nodes are connected
        for i in range(n - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                cost += G[u][v].get('weight', 1)  # default weight=1 if not set
            else:
                valid = False
                break

        # Update best path if valid and cheaper
        if valid and cost < min_cost:
            min_cost = cost
            best_path = path

    return best_path


def hamiltonian_path_heuristic(G: nx.Graph) -> list[int]:
    # TODO
    return None

def build_match_graph(kp_descs: list[np.ndarray], n: int) -> nx.Graph:
    match_graph = {x: [] for x in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            bf = cv2.BFMatcher(cv2.NORM_L2)

            kp1, des1 = kp_descs[i]
            kp2, des2 = kp_descs[j]

            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = image_utils.david_loew_ratio_test(matches)
            
            if len(good_matches) < 4:
                continue
                    
            match_graph[i].append([j, len(good_matches)])
            
    G = nx.Graph()

    # Add edges with weights
    for node, neighbors in match_graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, neighbor, weight=-weight)
            
    return G


def find_order(images: list[cv2.Mat]) -> list[int]:
    kp_descs = [feature_extraction.SIFT(img, nfeatures=1000) for img in images]
    
    G = build_match_graph(kp_descs, len(images))
    
    if len(images) < 10:
        return hamiltonian_path_brute_force(G)
    else:
        return hamiltonian_path_heuristic(G)