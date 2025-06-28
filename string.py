# ═══════════════════════════════════════════════════════════════════════════════
# LONGEST COMMON SUBSEQUENCE & PALINDROMIC SUBSEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

def lcs(a, b):
    """
    Compute the longest common subsequence (not necessarily contiguous) of a and b.
    Time: O(len(a) * len(b)), Space: O(len(a) * len(b))
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    i, j = n, m
    res = []
    while i and j:
        if dp[i][j] == dp[i-1][j]:
            i -= 1
        elif dp[i][j] == dp[i][j-1]:
            j -= 1
        else:
            res.append(a[i-1])
            i -= 1
            j -= 1
    return ''.join(reversed(res))

def lps(s):
    """
    Compute the longest palindromic subsequence of s by finding LCS of s and its reverse.
    Time: O(n^2), Space: O(n^2)
    """
    return lcs(s, s[::-1])

# ═══════════════════════════════════════════════════════════════════════════════
# BASIC PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def lps(s):
    """
    For each position i, compute the longest proper prefix of s[:i+1]
    that is also a suffix ending at i (the LPS array).
    Time: O(n), Space: O(n)
    """
    n = len(s)
    lps = [0] * n
    length = 0
    for i in range(1, n):
        while length and s[i] != s[length]:
            length = lps[length - 1]
        if s[i] == s[length]:
            length += 1
        lps[i] = length
    return lps

def zfn(s):
    """
    For each position i, compute how many characters from s[i:]
    match the start of s exactly.
    Time: O(n), Space: O(n)
    """
    n = len(s)
    Z = [0] * n
    left = right = 0
    for i in range(1, n):
        if i < right:
            Z[i] = min(right - i, Z[i - left])
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        if i + Z[i] > right:
            left, right = i, i + Z[i]
    Z[0] = n
    return Z

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING
# ═══════════════════════════════════════════════════════════════════════════════

def kmp(text, pattern):
    """
    Find all start positions where pattern appears in text,
    using the LPS array to skip characters smartly.
    Time: O(len(text) + len(pattern))
    """
    l = lps(pattern)
    result = []
    j = 0
    for i, c in enumerate(text):
        while j and c != pattern[j]:
            j = l[j - 1]
        if c == pattern[j]:
            j += 1
        if j == len(pattern):
            result.append(i - j + 1)
            j = l[j - 1]
    return result

class AhoCorasick:
    """
    Build a machine to find multiple patterns in one pass through text.
    Time: O(total pattern length + len(text) + number of matches)
    """
    def __init__(self, patterns):
        self.next = [{}]
        self.link = [-1]
        self.out = [[]]
        from collections import deque
        # Build trie
        for idx, pat in enumerate(patterns):
            node = 0
            for c in pat:
                if c not in self.next[node]:
                    self.next[node][c] = len(self.next)
                    self.next.append({})
                    self.link.append(-1)
                    self.out.append([])
                node = self.next[node][c]
            self.out[node].append(idx)
        # Build fallback links
        q = deque()
        for c, v in self.next[0].items():
            self.link[v] = 0
            q.append(v)
        while q:
            u = q.popleft()
            for c, v in self.next[u].items():
                q.append(v)
                j = self.link[u]
                while j != -1 and c not in self.next[j]:
                    j = self.link[j]
                self.link[v] = self.next[j].get(c, 0)
                self.out[v] += self.out[self.link[v]]

    def search(self, text):
        """
        Walk the text through the automaton, returning
        (position, pattern) for each found pattern.
        Time: O(len(text) + matches)
        """
        node = 0
        results = []
        for i, c in enumerate(text):
            while node and c not in self.next[node]:
                node = self.link[node]
            node = self.next[node].get(c, 0)
            for pat_idx in self.out[node]:
                results.append((i, pat_idx))
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# SUFFIX STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

def suffix_array(s):
    """
    Build an array of all suffix start positions sorted lexicographically.
    Time: O(n log n), Space: O(n)
    """
    n = len(s)
    sa = list(range(n))
    rank = [ord(c) for c in s] + [-1]
    tmp = [0] * n
    k = 1
    while k < n:
        sa.sort(key=lambda i: (rank[i], rank[i+k] if i+k<n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            prev, cur = sa[i-1], sa[i]
            tmp[cur] = tmp[prev] + ((rank[prev], rank[prev+k]) < (rank[cur], rank[cur+k] if cur+k<n else -1))
        rank, k = tmp[:], k*2
    return sa

def lcp_array(s, sa):
    """
    Build LCP array: lcp[i] = longest common prefix of suffixes sa[i] and sa[i+1].
    Time: O(n), Space: O(n)
    """
    n = len(s)
    rank = [0]*n
    for i, pos in enumerate(sa):
        rank[pos] = i
    lcp = [0]*(n-1)
    h = 0
    for i in range(n):
        if rank[i]:
            j = sa[rank[i]-1]
            while i+h<n and j+h<n and s[i+h]==s[j+h]:
                h += 1
            lcp[rank[i]-1] = h
            if h:
                h -= 1
    return lcp

class SuffixAutomaton:
    """
    Suffix Automaton: compact structure representing all substrings of a string.
    Time: O(n), Space: O(n)
    """

    def __init__(self, s):
        self.next = [{}]
        self.link = [-1]
        self.length = [0]
        last = 0
        for c in s:
            cur = len(self.next)
            self.next.append({})
            self.length.append(self.length[last] + 1)
            self.link.append(0)
            p = last
            while p >= 0 and c not in self.next[p]:
                self.next[p][c] = cur
                p = self.link[p]
            if p != -1:
                q = self.next[p][c]
                if self.length[p] + 1 != self.length[q]:
                    clone = len(self.next)
                    self.next.append(self.next[q].copy())
                    self.length.append(self.length[p] + 1)
                    self.link.append(self.link[q])
                    while p >= 0 and self.next[p][c] == q:
                        self.next[p][c] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
                else:
                    self.link[cur] = q
            last = cur

    def count_distinct(self):
        """
        Count distinct substrings in the original string.
        Time: O(n)
        """
        return sum(self.length[i] - self.length[self.link[i]] for i in range(1, len(self.next)))

    def contains(self, s):
        """
        Check if the string s is a substring of the original string.
        Time: O(len(s))
        """
        state = 0
        for c in s:
            if c not in self.next[state]:
                return False
            state = self.next[state][c]
        return True

    def prepare_occurrence_count(self):
        """
        Precompute the number of times each substring occurs.
        Time: O(n)
        """
        self.occurrence = [0] * len(self.next)
        for i in range(len(self.next)):
            self.occurrence[i] = 0
        terminal = [0] * len(self.next)
        p = len(self.length) - 1
        while p != -1:
            terminal[p] = 1
            p = self.link[p]

        for i in range(len(self.next)):
            self.occurrence[i] = terminal[i]

        order = sorted(range(len(self.next)), key=lambda x: -self.length[x])
        for u in order:
            if self.link[u] != -1:
                self.occurrence[self.link[u]] += self.occurrence[u]

    def count_occurrences(self, s):
        """
        Count the number of occurrences of string s in the original string.
        Time: O(len(s))
        """
        state = 0
        for c in s:
            if c not in self.next[state]:
                return 0
            state = self.next[state][c]
        return self.occurrence[state]

    def lcs(self, t):
        """
        Find the longest common substring between original and another string t.
        Time: O(len(t))
        """
        state = 0
        length = 0
        best_len = 0
        best_pos = 0

        for i, c in enumerate(t):
            while state and c not in self.next[state]:
                state = self.link[state]
                length = self.length[state]
            if c in self.next[state]:
                state = self.next[state][c]
                length += 1
            if length > best_len:
                best_len = length
                best_pos = i

        return t[best_pos - best_len + 1:best_pos + 1]

    def kth_substring(self, k):
        """
        Return the k-th lexicographically smallest substring (1-based).
        Time: O(n * alphabet)
        """
        if not hasattr(self, 'sub_count'):
            self.sub_count = [0] * len(self.next)

            def dfs(u):
                res = 1
                for c in sorted(self.next[u]):
                    res += dfs(self.next[u][c])
                self.sub_count[u] = res
                return res

            dfs(0)

        result = ""
        state = 0
        k += 1

        while k > 0:
            for c in sorted(self.next[state]):
                child = self.next[state][c]
                if self.sub_count[child] >= k:
                    result += c
                    state = child
                    k -= 1
                    break
                else:
                    k -= self.sub_count[child]
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# LONGEST PALINDROMIC SUBSTRING
# ═══════════════════════════════════════════════════════════════════════════════

def manacher(s):
    """
    Find the longest palindromic substring in s.
    Time: O(n), Space: O(n)
    """
    # Transform s to avoid even/odd case checks
    t = '^#' + '#'.join(s) + '#$'
    n = len(t)
    p = [0] * n
    center = right = 0
    for i in range(1, n-1):
        mirror = 2*center - i
        if i < right:
            p[i] = min(right - i, p[mirror])
        # Expand around center i
        while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > right:
            center, right = i, i + p[i]
    # Find max palindrome
    length, idx = max((p[i], i) for i in range(n))
    start = (idx - length) // 2
    return s[start:start + length]

# ═══════════════════════════════════════════════════════════════════════════════
# HASHING
# ═══════════════════════════════════════════════════════════════════════════════

class RollingHash:
    """
    Efficient rolling hash for lowercase letters using polynomial hashing.
    Time: O(n) preprocessing, O(1) query
    Space: O(n)
    """
    def __init__(self, s, base=31, mod=10**9 + 9):
        self.mod = mod
        self.base = base
        n = len(s)
        self.h = [0] * (n + 1)
        self.p = [1] * (n + 1)
        for i, c in enumerate(s):
            x = ord(c) - ord('a') + 1
            self.h[i + 1] = (self.h[i] + x * self.p[i]) % mod
            self.p[i + 1] = (self.p[i] * base) % mod

    def hash(self, l, r):
        """Hash of s[l:r] (0-based, r exclusive)"""
        return ((self.h[r] - self.h[l]) * pow(self.p[l], -1, self.mod)) % self.mod


# ═══════════════════════════════════════════════════════════════════════════════
# LYNDON FACTORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def factors(s):
    """
    Decompose s into a sequence of non-increasing “Lyndon words” (Duval's algorithm).
    Time: O(n), Space: O(n)
    """
    n = len(s)
    i = 0
    factors = []
    while i < n:
        j = i + 1
        k = i
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i
            else:
                k += 1
            j += 1
        while i <= k:
            factors.append(s[i:i + j - k])
            i += j - k
    return factors

# ═══════════════════════════════════════════════════════════════════════════════
# TRIE
# ═══════════════════════════════════════════════════════════════════════════════

class Trie:
    """
    Trie (Prefix Tree) for storing and querying strings efficiently.
    Operations are performed over lowercase letters by default.
    """

    def __init__(self, *words):
        """
        Initialize an empty Trie and insert any initial words.
        """
        self.root = {}
        for word in words:
            self.add(word)

    def add(self, word):
        """
        Insert a word into the trie.
        Time: O(L), Space: O(L)
        """
        current_dict = self.root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict["_end_"] = True

    def __contains__(self, word):
        """
        Check if a word is in the trie.
        Time: O(L)
        """
        current_dict = self.root
        for letter in word:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return "_end_" in current_dict

    def __delitem__(self, word):
        """
        Remove a word from the trie if it exists.
        Time: O(L)
        """
        current_dict = self.root
        nodes = [current_dict]
        for letter in word:
            current_dict = current_dict[letter]
            nodes.append(current_dict)
        del current_dict["_end_"]

    def prefix(self, prefix):
        """
        Return True if any word in the trie starts with the given prefix.
        Time: O(L)
        """
        current_dict = self.root
        for letter in prefix:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return True