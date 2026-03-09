### reference only
from collections import deque

def causal_induced_preference_verification(user_id, E_u_base, E_u_ext, C_y):
    E_u_base = set(E_u_base)
    E_u_ext = set(E_u_ext)
    E_u_c = E_u_base.union(E_u_ext)
    C_y_u = {
        node: [child for child in C_y.get(node, []) if child in E_u_c]
        for node in E_u_c
    }
    S_u = set(E_u_base)
    Q = deque(E_u_base)
    while Q:
        p = Q.popleft()
        for q in C_y_u.get(p, []):
            if q not in S_u:
                S_u.add(q)
                Q.append(q)
    T_u_y = {(user_id, p) for p in S_u}
    return T_u_y
