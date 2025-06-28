#═══════════════════════════════════════════════════════════════════════════════
# BASIC DP ON TREES
#═══════════════════════════════════════════════════════════════════════════════

def order(tree, root=0):
    """
    DFS order, parent, and depth array for a tree
    Time: O(V + E), Space: O(V)
    """
    n = len(tree)
    vis = [False] * n
    vis[root] = True
    stack = [root]
    parent = [-1] * n
    depth = [0] * n
    order = [root]

    while stack:
        current = stack.pop()
        for child in tree[current]:
            if not vis[child]:
                vis[child] = True
                stack.append(child)
                order.append(child)
                parent[child] = current
                depth[child] = depth[current] + 1

    return order, parent, depth

def size(tree, root=0):
    """
    Computes the size of the subtree rooted at each node
    Time: O(V + E), Space: O(V)
    """
    order, parent, _ = order(tree, root)
    n = len(tree)
    size = [1] * n
    for node in order[:0:-1]:
        size[parent[node]] += size[node]
    return size

#═══════════════════════════════════════════════════════════════════════════════
# TREE DIAMETER
#═══════════════════════════════════════════════════════════════════════════════

def diameter(tree):
    """
    Finds the two endpoints of the diameter of a tree
    Time: O(V + E), Space: O(V)
    """
    def dfs(tree, root):
        n = len(tree)
        vis = [False] * n
        dist = [0] * n
        stack = [root]
        vis[root] = True

        while stack:
            node = stack.pop()
            for neighbor in tree[node]:
                if not vis[neighbor]:
                    vis[neighbor] = True
                    dist[neighbor] = dist[node] + 1
                    stack.append(neighbor)

        return max(((dist[i], i) for i in range(n)))[1]

    u = dfs(tree, 0)
    v = dfs(tree, u)
    return u, v

#═══════════════════════════════════════════════════════════════════════════════
# EULER TOUR FOR SUBTREE QUERIES
#═══════════════════════════════════════════════════════════════════════════════

def tour(tree, root=0):
    """
    Euler tour to get in/out times for subtree queries
    Time: O(V + E), Space: O(V)
    """
    n = len(tree)
    pointer = [0] * n
    stack = [root]
    path = []

    while stack:
        node = stack[-1]
        if pointer[node] < len(tree[node]):
            next = tree[node][pointer[node]]
            pointer[node] += 1
            stack.append(next)
        else:
            path.append(stack.pop())

    time = [[-1, -1] for i in range(n)]
    for j in range(2 * n - 1):
        if time[path[j]][0] == -1:
            time[path[j]][0] = j
        time[path[j]][1] = j

    return time

#═══════════════════════════════════════════════════════════════════════════════
# LOWEST COMMON ANCESTOR
#═══════════════════════════════════════════════════════════════════════════════

class rmq:
    """
    Sparse Table for range queries
    Time: O(N * log N) preprocessing, O(1) query
    Space: O(N * log N)
    """
    def __init__(self, data, func=min):
        self.func = func
        self._data = _data = [list(data)]
        i, n = 1, len(_data[0])
        while 2 * i <= n:
            prev = _data[-1]
            _data.append([func(prev[j], prev[j + i]) for j in range(n - 2 * i + 1)])
            i <<= 1
 
    def query(self, begin, end):
        depth = (end - begin).bit_length() - 1
        return self.func(
            self._data[depth][begin],
            self._data[depth][end - (1 << depth)]
        )
class lca:
    """
    Lowest Common Ancestor using Euler Tour + RMQ (incorrect version)
    Time: O(V * log V) preprocessing, O(1) per query
    Space: O(V * log V)
    """
    def __init__(self, graph, root=0):
        self.time = [-1] * len(graph)
        self.path = [-1] * len(graph)
        P = [-1] * len(graph)
        t = -1
        dfs = [root]
 
        while dfs:
            node = dfs.pop()
            self.path[t] = P[node]
            self.time[node] = t = t + 1
            for nei in graph[node]:
                if self.time[nei] == -1:
                    P[nei] = node
                    dfs.append(nei)
 
        self.rmq = rmq(self.time[node] for node in self.path)
 
    def __call__(self, a, b):
        if a == b:
            return a
        a = self.time[a]
        b = self.time[b]
        if a > b:
            a, b = b, a
        return self.path[self.rmq.query(a, b)]
    
#═══════════════════════════════════════════════════════════════════════════════
# BINARY LIFTING ON TREE
#═══════════════════════════════════════════════════════════════════════════════

class binary_lift:
    """Binary lifting for LCA, kth ancestor, distance, and path queries."""
    def __init__(self, graph, data=(), f=min, root=0):
        n = len(graph)

        parent = [-1] * (n)
        depth = self.depth = [-1] * n
        bfs = [root]
        depth[root] = 0
        for node in bfs:
            for nei in graph[node]:
                if depth[nei] == -1:
                    parent[nei] = node
                    depth[nei] = depth[node] + 1
                    bfs.append(nei)

        data = self.data = [list(data) if data else [0]*n]
        parent = self.parent = [parent]
        self.f = f

        for _ in range(max(depth).bit_length()):
            old_data = data[-1]
            old_parent = parent[-1]

            data.append([f(val, old_data[p]) if p != -1 else val for val, p in zip(old_data, old_parent)])
            parent.append([old_parent[p] if p != -1 else -1 for p in old_parent])

    def lca(self, a, b):
        depth = self.depth
        parent = self.parent

        if depth[a] < depth[b]:
            a, b = b, a

        d = depth[a] - depth[b]
        for i in range(d.bit_length()):
            if (d >> i) & 1:
                a = parent[i][a]

        for i in range(depth[a].bit_length() - 1, -1, -1):
            if parent[i][a] != parent[i][b]:
                a = parent[i][a]
                b = parent[i][b]

        if a != b:
            return parent[0][a]
        else:
            return a

    def distance(self, a, b):
        return self.depth[a] + self.depth[b] - 2 * self.depth[self.lca(a, b)]

    def kth_ancestor(self, a, k):
        parent = self.parent
        if self.depth[a] < k:
            return -1
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                a = parent[i][a]
                if a == -1:
                    return -1
        return a

    def __call__(self, a, b):
        depth = self.depth
        parent = self.parent
        data = self.data
        f = self.f

        c = self.lca(a, b)
        val = data[0][c]
        for x, dist in ((a, depth[a] - depth[c]), (b, depth[b] - depth[c])):
            for i in range(dist.bit_length()):
                if (dist >> i) & 1:
                    val = f(val, data[i][x])
                    x = parent[i][x]

        return val
    
#═══════════════════════════════════════════════════════════════════════════════
# CENTROID DECOMPOSITION OF TREE
#═══════════════════════════════════════════════════════════════════════════════

def centroid(graph):
    """ 
    Perform centroid decomposition on a tree.
    This generator:
    1. Roots the tree at its centroid (modifies graph)
    2. Yields the centroid node
    3. Removes the centroid from the graph
    4. Recurses on the remaining forest
    
    The graph is updated so it remains rooted at the yielded centroid.
    Time: O(V * log V).
    """
    n = len(graph)
    
    bfs = [n - 1]
    for node in bfs:
        bfs += graph[node]
        for nei in graph[node]:
            graph[nei].remove(node)
    
    size = [0] * n
    for node in reversed(bfs):
        size[node] = 1 + sum(size[child] for child in graph[node])
 
    def reroot_centroid(root):
        N = size[root]
        while True:
            for child in graph[root]:
                if size[child] > N // 2:
                    size[root] = N - size[child]
                    graph[root].remove(child)
                    graph[child].append(root)
                    root = child
                    break
            else:
                return root
        
    bfs = [n - 1]
    for node in bfs:
        centroid = reroot_centroid(node)
        bfs += graph[centroid]
        yield centroid