class SegTree2D:
    def __init__(self, matrix, default=0, func=max):
        self._func = func
        self._default = default

        self.n = len(matrix)
        self.m = len(matrix[0]) if self.n else 0

        self._n = 1 << (self.n - 1).bit_length()
        self._m = 1 << (self.m - 1).bit_length()

        size_n, size_m = 2 * self._n, 2 * self._m
        self.tree = [[default] * size_m for _ in range(size_n)]

        for i in range(self.n):
            for j in range(self.m):
                self.tree[i + self._n][j + self._m] = matrix[i][j]

        for i in range(self.n):
            row = self.tree[i + self._n]
            for j in reversed(range(1, self._m)):
                row[j] = func(row[2 * j], row[2 * j + 1])

        for i in reversed(range(1, self._n)):
            for j in range(2 * self._m):
                self.tree[i][j] = func(self.tree[2 * i][j], self.tree[2 * i + 1][j])

    def update(self, x, y, value):
        """Set matrix[x][y] = value"""
        i, j = x + self._n, y + self._m
        self.tree[i][j] = value

        jj = j
        while jj > 1:
            jj //= 2
            self.tree[i][jj] = self._func(self.tree[i][2 * jj], self.tree[i][2 * jj + 1])

        ii = i
        while ii > 1:
            ii //= 2
            jj = j
            while jj >= 1:
                self.tree[ii][jj] = self._func(self.tree[2 * ii][jj], self.tree[2 * ii + 1][jj])
                jj //= 2

    def query(self, x1, y1, x2, y2):
        """Query func in matrix[x1:x2][y1:y2] (half-open range)"""
        res = self._default
        x1 += self._n
        x2 += self._n
        while x1 < x2:
            if x1 % 2 == 1:
                res = self._func(res, self._query_y(x1, y1, y2))
                x1 += 1
            if x2 % 2 == 1:
                x2 -= 1
                res = self._func(res, self._query_y(x2, y1, y2))
            x1 //= 2
            x2 //= 2
        return res

    def _query_y(self, i, y1, y2):
        res = self._default
        y1 += self._m
        y2 += self._m
        while y1 < y2:
            if y1 % 2 == 1:
                res = self._func(res, self.tree[i][y1])
                y1 += 1
            if y2 % 2 == 1:
                y2 -= 1
                res = self._func(res, self.tree[i][y2])
            y1 //= 2
            y2 //= 2
        return res