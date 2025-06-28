import collections
import heapq

#═══════════════════════════════════════════════════════════════════════════════
# BASIC GRAPH TRAVERSAL
#═══════════════════════════════════════════════════════════════════════════════

def dfs(graph, start):
    """Depth-first search traversal. Time: O(V + E), Space: O(V)"""
    visited = [False] * len(graph)
    stack = [start]
    visited[start] = True
    
    while stack:
        current = stack.pop()
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return

def bfs(graph, start):
    """Breadth-first search traversal. Time: O(V + E), Space: O(V)"""
    visited = [False] * len(graph)
    queue = collections.deque([start])
    visited[start] = True
    
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return

#═══════════════════════════════════════════════════════════════════════════════
# CYCLE DETECTION
#═══════════════════════════════════════════════════════════════════════════════

def undCycle(graph):
    """Cycle detection in an undirected graph using iterative DFS.
    Time: O(V + E), Space: O(V)
    """
    n = len(graph)
    vis = [False] * n
    par = [-1] * n

    for node in range(n):
        if not vis[node]:
            stack = [node]
            vis[node] = True
            while stack:
                parent = stack.pop()
                for child in graph[parent]:
                    if not vis[child]:
                        vis[child] = True
                        par[child] = parent
                        stack.append(child)
                    elif par[parent] != child:
                        return True
    return False

def dirCycle(graph):
    """Cycle detection in a directed graph using DFS with recursion tracking.
    Time: O(V + E), Space: O(V)
    """
    n = len(graph)
    vis = [False] * n
    rec = [False] * n

    def dfs(node):
        vis[node] = rec[node] = True
        for neighbor in graph[node]:
            if not vis[neighbor]:
                if dfs(neighbor):
                    return True
            elif rec[neighbor]:
                return True
        rec[node] = False
        return False

    for i in range(n):
        if not vis[i]:
            if dfs(i):
                return True
    return False

#═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGICAL SORT
#═══════════════════════════════════════════════════════════════════════════════

def order(graph):
    """Topological sort of a directed acyclic graph (DAG). Time: O(V + E), Space: O(V)"""
    n = len(graph)
    indeg = [0] * n

    for u in range(n):
        for v in graph[u]:
            indeg[v] += 1

    stack = [u for u in range(n) if indeg[u] == 0]
    order = []

    while stack:
        current = stack.pop()
        order.append(current)
        for neighbor in graph[current]:
            indeg[neighbor] -= 1
            if indeg[neighbor] == 0:
                stack.append(neighbor)

    return order

#═══════════════════════════════════════════════════════════════════════════════
# SHORTEST PATHS
#═══════════════════════════════════════════════════════════════════════════════

def dijkstra(graph, start):
    """Dijkstra's algorithm for single-source shortest paths in a weighted graph with non-negative edges.
    Time: O((V + E) * log V), Space: O(V)
    """
    heap = [(0, start)]
    dist = [-1] * len(graph)

    while heap:
        distance, current = heapq.heappop(heap)
        if dist[current] == -1:
            dist[current] = distance
            for neighbor, weight in graph[current]:
                if dist[neighbor] == -1:
                    heapq.heappush(heap, (distance + weight, neighbor))

    return dist

def bellmanFord(edges, n, start):
    """Bellman-Ford algorithm for single-source shortest paths with negative edge support.
    Time: O(V * E), Space: O(V)
    """
    inf=1<<32
    dist = [(inf)] * n
    dist[start] = 0

    for i in range(n - 1):
        for u, v, w in edges:
            dist[v] = min(dist[v], dist[u] + w)

    return dist

def floydWarshall(dp):
    """Floyd-Warshall algorithm for all-pairs shortest paths.
    Time: O(V^3), Space: O(V^2)
    """
    n = len(dp)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j])

    return dp

#═══════════════════════════════════════════════════════════════════════════════
# MINIMUM SPANNING TREE
#═══════════════════════════════════════════════════════════════════════════════

def kruskal(edges, n):
    """Kruskal's algorithm for Minimum Spanning Tree (MST) using Union-Find.
    Time: O(E * log E), Space: O(V)
    """
    d = dsu(n)
    edges.sort(key=lambda x: x[2])
    count, weight = 0, 0

    for u, v, w in edges:
        if d.find(u) != d.find(v):
            d.join(u, v)
            count += 1
            weight += w

    return weight if count == n - 1 else -1

def prim(graph):
    """Prim's algorithm for Minimum Spanning Tree (MST) using a priority queue.
    Time: O(E * log V), Space: O(V)
    """
    heap = [(0, 0)]
    vis = [False] * len(graph)
    ans = 0

    while heap:
        distance, current = heapq.heappop(heap)
        if not vis[current]:
            vis[current] = True
            ans += distance
            for neighbor, weight in graph[current]:
                if not vis[neighbor]:
                    heapq.heappush(heap, (weight, neighbor))

    return ans

#═══════════════════════════════════════════════════════════════════════════════
# STRONGLY CONNECTED COMPONENTS
#═══════════════════════════════════════════════════════════════════════════════

def scc(graph):
    """Kosaraju's algorithm to find Strongly Connected Components (SCCs).
    Time: O(V + E), Space: O(V + E)
    """
    n, stack, vis, rev = len(graph), [], [0] * n, [[] for i in range(n)]

    def dfs(parent):
        vis[parent] = 1
        for child in graph[parent]:
            rev[child].append(parent)
            if not vis[child]:
                dfs(child)
        stack.append(parent)
    
    for j in range(n):
        if not vis[j]:
            dfs(j)

    scc = []
    vis = [0] * n

    while stack:
        x = stack.pop()
        if not vis[x]:
            scc.append([])
            stk = [x]
            vis[x] = 1
            while stk:
                parent = stk.pop()
                scc[-1].append(parent)
                for child in rev[parent]:
                    if not vis[child]:
                        stk.append(child)
                        vis[child] = 1

    return scc

def scc(graph):
    """Tarjan's algorithm (iterative) to find Strongly Connected Components (SCCs).
    Time: O(V + E), Space: O(V)
    """
    scc, s, p = [], [], []
    depth = [0] * len(graph)

    stack = list(range(len(graph)))
    while stack:
        node = stack.pop()
        if node < 0:
            d = depth[~node] - 1
            if p[-1] > d:
                scc.append(s[d:])
                del s[d:], p[-1]
                for node in scc[-1]:
                    depth[node] = -1
        elif depth[node] > 0:
            while p[-1] > depth[node]:
                p.pop()
        elif depth[node] == 0:
            s.append(node)
            p.append(len(s))
            depth[node] = len(s)
            stack.append(~node)
            stack.extend(graph[node])

    return scc[::-1]

#═══════════════════════════════════════════════════════════════════════════════
# BI-CONNECTED COMPONENTS
#═══════════════════════════════════════════════════════════════════════════════

def bridges(graph):
    """
    Find all bridges in the undirected graph.
    Returns a list of edges (u, v) that are bridges.
    Time: O(V + E), Space: O(V)
    """
    n = len(graph)
    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    timer = 0
    bridges = []

    def rec(u, p=-1):
        nonlocal timer
        visited[u] = True
        disc[u] = low[u] = timer
        timer += 1

        for v in graph[u]:
            if v == p:
                continue
            if visited[v]:
                low[u] = min(low[u], disc[v])
            else:
                rec(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))

    for i in range(n):
        if not visited[i]:
            rec(i)

    return bridges

def articul(graph):
    """
    Find all articulation points in the undirected graph.
    Returns a list of node indices that are articulation points.
    Time: O(V + E), Space: O(V)
    """
    n = len(graph)
    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    timer = 0
    articul = set()

    def rec(u, p=-1):
        nonlocal timer
        visited[u] = True
        disc[u] = low[u] = timer
        timer += 1
        children = 0

        for v in graph[u]:
            if v == p:
                continue
            if visited[v]:
                low[u] = min(low[u], disc[v])
            else:
                children += 1
                rec(v, u)
                low[u] = min(low[u], low[v])
                if (p == -1 and children > 1) or (p != -1 and low[v] >= disc[u]):
                    articul.add(u)

    for i in range(n):
        if not visited[i]:
            rec(i)

    return articul

#═══════════════════════════════════════════════════════════════════════════════
# BIPARTITE, MATCHING & FLOWS
#═══════════════════════════════════════════════════════════════════════════════

def bipartite(graph):
    """
    Check if an undirected graph is bipartite using DFS.
    Time: O(V + E), Space: O(V)
    """
    n = len(graph)
    color = [-1] * n

    for i in range(n):
        if color[i] == -1:
            stack = [i]
            color[i] = 0

            while stack:
                u = stack.pop()
                for v in graph[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        return False

    return True

def hopcroftkarp(graph, n, m):
    """
    Maximum-cardinality bipartite matching using Hopcroft-Karp algorithm.
    graph: adjacency list for left side (size n), where graph[u] contains neighbors on right side [0..m-1]
    n: number of nodes on the left side
    m: number of nodes on the right side
    match1: list of matches for left side nodes (-1 if unmatched)
    match2: list of matches for right side nodes (-1 if unmatched)
    Time: O(E * sqrt(V)), Space: O(V + E)
    """
    assert n == len(graph)

    match1 = [-1] * n
    match2 = [-1] * m

    for u in range(n):
        for v in graph[u]:
            if match2[v] == -1:
                match1[u] = v
                match2[v] = u
                break

    while True:
        bfs = [u for u in range(n) if match1[u] == -1]
        depth = [-1] * n
        for u in bfs:
            depth[u] = 0

        found = False
        for u in bfs:
            for v in graph[u]:
                next_u = match2[v]
                if next_u == -1:
                    found = True
                    break
                if depth[next_u] == -1:
                    depth[next_u] = depth[u] + 1
                    bfs.append(next_u)
            if found:
                break
        if not found:
            break

        pointer = [len(graph[u]) for u in range(n)]
        stack = [u for u in range(n) if depth[u] == 0]

        while stack:
            u = stack[-1]
            while pointer[u]:
                pointer[u] -= 1
                v = graph[u][pointer[u]]
                next_u = match2[v]
                if next_u == -1:
                    while v != -1:
                        u = stack.pop()
                        match2[v], match1[u], v = u, v, match1[u]
                    break
                elif depth[u] + 1 == depth[next_u]:
                    stack.append(next_u)
                    break
            else:
                stack.pop()

    return match1, match2

def hungarian(cost):
    """
    Hungarian algorithm for maximum/minimum-cost perfect matching (Kuhn-Munkres algorithm).
    Time: O(n^3), Space: O(n^2)
    """
    n = len(cost)
    for i in range(n):
        for j in range(n):
            cost[i][j] = -cost[i][j]

    match = [0]
    xy = [-1] * n
    yx = [-1] * n
    lx = [max(row) for row in cost]
    ly = [0] * n
    slack = [0] * n
    slackX = [0] * n
    prev = [0] * n
    inTreeX = [False] * n
    inTreeY = [False] * n

    def add_tree(x, prevX):
        inTreeX[x] = True
        prev[x] = prevX
        for y in range(n):
            temp = lx[x] + ly[y] - cost[x][y]
            if temp < slack[y]:
                slack[y] = temp
                slackX[y] = x

    def update_labels():
        delta = min(slack[y] for y in range(n) if not inTreeY[y])
        for i in range(n):
            if inTreeX[i]:
                lx[i] -= delta
        for j in range(n):
            if inTreeY[j]:
                ly[j] += delta
            else:
                slack[j] -= delta

    def augment():
        if match[0] == n:
            return

        root = next(i for i in range(n) if xy[i] == -1)
        q = collections.deque([root])
        prev[root] = -2
        inTreeX[root] = True

        for y in range(n):
            slack[y] = lx[root] + ly[y] - cost[root][y]
            slackX[y] = root

        while True:
            while q:
                x = q.popleft()
                for y in range(n):
                    if cost[x][y] == lx[x] + ly[y] and not inTreeY[y]:
                        if yx[y] == -1:
                            return x, y
                        inTreeY[y] = True
                        q.append(yx[y])
                        add_tree(yx[y], x)

            update_labels()
            for y in range(n):
                if not inTreeY[y] and slack[y] == 0:
                    if yx[y] == -1:
                        return slackX[y], y
                    inTreeY[y] = True
                    if not inTreeX[yx[y]]:
                        q.append(yx[y])
                        add_tree(yx[y], slackX[y])

    while match[0] < n:
        for i in range(n):
            inTreeX[i] = inTreeY[i] = False
        x, y = augment()
        match[0] += 1
        cx, cy = x, y
        while cx != -2:
            ty = xy[cx]
            xy[cx] = cy
            yx[cy] = cx
            cx, cy = prev[cx], ty

    return abs(sum(cost[i][xy[i]] for i in range(n)))

def dinic(graph, s, t):
    """
    Computes the maximum flow from source `s` to sink `t` in a graph using Dinic's algorithm.
    Time: O(V^2 * E), Space: O(V + E)
    """
    inf = 1<<60
    n = len(graph)
    level = [0] * n
    ptr = [0] * n
    q = [0] * n

    def bfs():
        nonlocal level
        level = [0] * n
        q[0] = s
        level[s] = 1
        qi, qe = 0, 1
        while qi < qe:
            u = q[qi]
            qi += 1
            for v, _, cap, flow in graph[u]:
                if not level[v] and flow < cap:
                    level[v] = level[u] + 1
                    q[qe] = v
                    qe += 1
        return level[t] != 0

    def dfs(u, pushed):
        if u == t or not pushed:
            return pushed
        for i in range(ptr[u], len(graph[u])):
            v, rev, cap, flow = graph[u][i]
            if level[v] == level[u] + 1 and flow < cap:
                tr = dfs(v, min(pushed, cap - flow))
                if tr:
                    graph[u][i][3] += tr
                    graph[v][rev][3] -= tr
                    return tr
            ptr[u] += 1
        return 0

    flow = 0
    while bfs():
        ptr = [0] * n
        pushed = dfs(s, inf)
        while pushed:
            flow += pushed
            pushed = dfs(s, inf)
    return flow

#═══════════════════════════════════════════════════════════════════════════════
# EULER WALK
#═══════════════════════════════════════════════════════════════════════════════

def fleury(graph, root=0):
    """
    Fleury's Algorithm for finding an Eulerian path or circuit in an undirected graph.
    Time: O(E), Space: O(V)
    """
    n = len(graph)
    pointer = [0] * n
    stack = [root]
    path = []

    while stack:
        node = stack[-1]
        if pointer[node] < len(graph[node]):
            next = graph[node][pointer[node]]
            pointer[node] += 1
            stack.append(next)
        else:
            path.append(stack.pop())

    return path

#═══════════════════════════════════════════════════════════════════════════════
# Disjoint Set Union
#═══════════════════════════════════════════════════════════════════════════════

class dsu:
    """Disjoint Set Union (Union-Find) with path compression and union by size."""
    def __init__(self, n):
        self.size = [1] * n
        self.parent = list(range(n))
        self.length = n

    def find(self, a):
        b = a
        while a != self.parent[a]:
            a = self.parent[a]
        while b != a:
            b, self.parent[b] = self.parent[b], a
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.size[a] += self.size[b]
            self.parent[b] = a
            self.length -= 1