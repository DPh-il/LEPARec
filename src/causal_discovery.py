### reference only
from collections import deque

def creates_cycle(current_edges, new_edge):
    adj = {}
    for u, v in current_edges:
        if u not in adj: adj[u] = []
        adj[u].append(v)
    p, q = new_edge
    if p not in adj: adj[p] = []
    adj[p].append(q)
    visited_state = {}
    def dfs(node):
        if visited_state.get(node, 0) == 1:
            return True
        if visited_state.get(node, 0) == 2:
            return False
            
        visited_state[node] = 1
        for neighbor in adj.get(node, []):
            if dfs(neighbor):
                return True
        visited_state[node] = 2
        return False
    for node in list(adj.keys()):
        if visited_state.get(node, 0) == 0:
            if dfs(node):
                return True       
    return False

def bfs_based_global_preference_dag_discovery(E_c, llm_init_func, llm_bfs_func):
    R_c = set() 
    Q = deque()
    R_0 = llm_init_func(E_c)
    for p in R_0:
        Q.append(p)
    visited = set()
    while Q:
        p = Q.popleft()
        if p in visited:
            continue       
        C_y = (E_c, R_c)     
        N_plus_p = llm_bfs_func(p, E_c, C_y)      
        for q in N_plus_p:
            if not creates_cycle(R_c, (p, q)):
                R_c.add((p, q))
                Q.append(q)              
        visited.add(p)
    return (E_c, R_c)