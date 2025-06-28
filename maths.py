import math

#═══════════════════════════════════════════════════════════════════════════════
# BASIC ARITHMETIC & MODULAR OPERATIONS
#═══════════════════════════════════════════════════════════════════════════════

def power(a, b, mod):
    """Binary exponentiation: a^b % mod. Time: O(log b), Space: O(1)"""
    res = 1
    a %= mod
    while b > 0:
        if b & 1:
            res = (res * a) % mod
        a = (a * a) % mod
        b >>= 1
    return res

def add(a, b, mod):
    """Safe modular addition: (a + b) % mod. Handles negative numbers."""
    return ((a % mod) + (b % mod)) % mod

def sub(a, b, mod):
    """Safe modular subtraction: (a - b) % mod. Handles negative numbers."""
    return ((a % mod) - (b % mod)) % mod

def mul(a, b, mod):
    """Safe modular multiplication: (a * b) % mod. Handles negative numbers."""
    return ((a % mod) * (b % mod)) % mod

def inv(a, mod):
    """Modular inverse using Fermat's little theorem. mod must be prime."""
    return power(a, mod-2, mod)

def extgcd(a, b):
    """Extended GCD: returns (g, x, y) such that ax + by = gcd(a,b)."""
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extgcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y

def modinv(a, mod):
    """Modular inverse using extended GCD. Works for any coprime mod."""
    g, x, y = extgcd(a, mod)
    if g != 1:
        return -1  # no inverse exists
    return (x % mod + mod) % mod

def gcd(a, b):
    """Greatest Common Divisor of a and b. Uses Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Least Common Multiple of a and b."""
    return a // gcd(a, b) * b

#═══════════════════════════════════════════════════════════════════════════════
# PRIME TESTING & FACTORIZATION
#═══════════════════════════════════════════════════════════════════════════════

def isPrime(n):
    """Miller-Rabin primality test (deterministic for n < 2^64). Time: O(k log n)."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    # Check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    
    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        composite = True
        for _ in range(1, s):
            x = (x * x) % n
            if x == n-1:
                composite = False
                break
        if composite:
            return False
    return True

def pollard(n):
    """Pollard's Rho factorization. Returns a factor of n. Time: O(n^1/4) expected."""
    if n % 2 == 0:
        return 2
    if isPrime(n):
        return n
    
    def f(x, c, n):
        return (x*x + c) % n
        
    for c in range(1, 21):
        x = 2
        y = 2
        d = 1
        while d == 1:
            x = f(x, c, n)
            y = f(f(y, c, n), c, n)
            d = gcd(abs(x - y), n)
        if d > 1 and d < n:
            return d
    return n  # fallback

def factor(n):
    """Complete factorization of n using Pollard Rho + Miller-Rabin. Returns list of prime factors."""
    if n == 1:
        return []
    if isPrime(n):
        return [n]
    d = pollard(n)
    return factor(d) + factor(n // d)

#═══════════════════════════════════════════════════════════════════════════════
# PRIME FACTORIZATION METHODS
#═══════════════════════════════════════════════════════════════════════════════

def getPrimeFactorsMult(n):
    """Prime factorization using trial division. Returns list of (prime, exponent). Time: O(√n)."""
    factors = []
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            cnt = 0
            while temp % i == 0:
                cnt += 1
                temp //= i
            factors.append((i, cnt))
        i += 1
    if temp > 1:
        factors.append((temp, 1))
    return factors

def getPrimeFactorsDist(n):
    """Returns distinct prime factors of n. Time: O(√n)."""
    factors = []
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            factors.append(i)
            while temp % i == 0:
                temp //= i
        i += 1
    if temp > 1:
        factors.append(temp)
    return factors

def computeSPF(n):
    """Computes smallest prime factor (SPF) for numbers up to n. Time: O(n log log n)."""
    spf = list(range(n+1))
    for i in range(2, math.isqrt(n)+1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, n+1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def getPrimeFactorsMultSPF(x, spf):
    """Prime factorization using SPF array. Returns list of (prime, exponent). Time: O(log x)."""
    factors = []
    while x > 1:
        p = spf[x]
        cnt = 0
        while spf[x] == p:
            cnt += 1
            x //= p
        factors.append((p, cnt))
    return factors

def getPrimeFactorsDistSPF(x, spf):
    """Returns distinct prime factors using SPF array. Time: O(log x)."""
    factors = []
    while x > 1:
        p = spf[x]
        factors.append(p)
        while x % p == 0:
            x //= p
    return factors

def getPrimeFactorsLarge(n):
    """Prime factorization for large numbers using Pollard's Rho. Returns list of (prime, exponent)."""
    fct = factor(n)
    factors = {}
    for p in fct:
        factors[p] = factors.get(p, 0) + 1
    return factors

#═══════════════════════════════════════════════════════════════════════════════
# SIEVE & MULTIPLICATIVE FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════════

def linearSieve(n):
    """Linear sieve for primes, phi, mu, d (divisors), sigma (sum of divisors). Time: O(n)."""
    is_comp = [False] * (n+1)
    primes, phi, mu, d, sigma =[], [0] * (n + 1), [0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    phi[1], mu[1], d[1], sigma[1] = 1, 1, 1, 1

    for i in range(2, n+1):
        if not is_comp[i]:
            primes.append(i)
            phi[i] = i - 1
            mu[i] = -1
            d[i] = 2
            sigma[i] = i + 1
        
        for p in primes:
            if i * p > n:
                break
            ip = i * p
            is_comp[ip] = True
            
            if i % p == 0:
                phi[ip] = phi[i] * p
                mu[ip] = 0
                
                # Count powers of p in i
                temp = i
                cnt = 0
                while temp % p == 0:
                    cnt += 1
                    temp //= p
                d[ip] = d[i] // (cnt+1) * (cnt+2)
                # sigma(ip) = sigma(i) * (p^(cnt+2)-1) / (p^(cnt+1)-1)
                sigma[ip] = sigma[i] * (pow(p, cnt+2) - 1) // (pow(p, cnt+1) - 1)
                break
            else:
                phi[ip] = phi[i] * (p-1)
                mu[ip] = -mu[i]
                d[ip] = d[i] * 2
                sigma[ip] = sigma[i] * (p+1)
    
    return primes, phi, mu, d, sigma

#═══════════════════════════════════════════════════════════════════════════════
# COMBINATORICS
#═══════════════════════════════════════════════════════════════════════════════

def factorial(n, mod):
    """Precomputes factorials and inverse factorials up to n modulo mod. Time: O(n)."""
    fact = [1] * (n+1)
    invfact = [1] * (n+1)
    for i in range(1, n+1):
        fact[i] = fact[i-1] * i % mod
    invfact[n] = pow(fact[n], mod-2, mod)
    for i in range(n, 0, -1):
        invfact[i-1] = invfact[i] * i % mod
    return fact, invfact

def nCr(n, r, fact, invfact, mod):
    """Combination: n choose r modulo mod. Requires precomputed factorials."""
    if r < 0 or r > n:
        return 0
    return fact[n] * invfact[r] % mod * invfact[n-r] % mod

def nPr(n, r, fact, invfact, mod):
    """Permutation: nPr = n! / (n-r)! modulo mod. Requires precomputed factorials."""
    if r < 0 or r > n:
        return 0
    return fact[n] * invfact[n-r] % mod

def lucas(n, r, p, fact, invfact):
    """Lucas theorem for nCr mod p (p prime). Can precompute factorials for p."""
    if r == 0:
        return 1
    ni = n % p
    ri = r % p
    if ri > ni:
        return 0
    return nCr(ni, ri, fact, invfact, p) * lucas(n//p, r//p, p, fact, invfact) % p

def catalan(n, fact, invfact, mod):
    """Catalan number: C_n = (1/(n+1)) * C(2n, n) modulo mod."""
    return nCr(2*n, n, fact, invfact, mod) * pow(n+1, mod-2, mod) % mod

def stars(k, n, fact, invfact, mod):
    """Number of ways to put k identical items into n distinct boxes (non-negative)."""
    if k < 0 or n <= 0:
        return 0
    return nCr(k+n-1, n-1, fact, invfact, mod)

#═══════════════════════════════════════════════════════════════════════════════
# ADVANCED NUMBER THEORY
#═══════════════════════════════════════════════════════════════════════════════

def crt2(a1, m1, a2, m2):
    """Solves x ≡ a1 (mod m1), x ≡ a2 (mod m2). Returns (x, lcm(m1,m2)) or (-1,-1) if no solution."""
    g, x, y = extgcd(m1, m2)
    if (a2 - a1) % g != 0:
        return -1, -1
    lcm_val = m1 // g * m2
    res = a1 + (a2 - a1) // g * x % (m2 // g) * m1
    return res % lcm_val, lcm_val

def crt(a, m):
    """General CRT for multiple congruences. Requires pairwise coprime moduli."""
    res = a[0]
    mod = m[0]
    for i in range(1, len(a)):
        res, mod = crt2(res, mod, a[i], m[i])
        if res == -1:
            return -1
    return res

def bsgs(a, b, m):
    """Baby-step giant-step: solves a^x ≡ b (mod m). Returns smallest x or -1."""
    a %= m
    b %= m
    n = int(math.isqrt(m)) + 1
    vals = {}
    cur = 1
    for i in range(n):
        if cur not in vals:
            vals[cur] = i
        cur = (cur * a) % m
    
    an = pow(a, n, m)
    if gcd(an, m) != 1:  # Handle non-coprime case
        return -1
    inv_an = pow(an, -1, m)
    cur = b
    for i in range(n):
        if cur in vals:
            ans = i * n + vals[cur]
            if ans > 0:
                return ans
        cur = (cur * inv_an) % m
    return -1

def legendre(a, p):
    """Legendre symbol: 1 if quadratic residue, -1 if not, 0 if a ≡ 0 mod p."""
    res = pow(a, (p-1)//2, p)
    if res == p-1:
        return -1
    return res

def tonelli(n, p):
    """Tonelli-Shanks algorithm for modular square root. Returns x such that x^2 ≡ n mod p."""
    if legendre(n, p) != 1:
        return -1
    
    # Factor p-1 as Q * 2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    if S == 1:
        return pow(n, (p+1)//4, p)
    
    # Find quadratic non-residue
    z = 2
    while legendre(z, p) != -1:
        z += 1
    
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q+1)//2, p)
    
    while t != 1:
        # Find smallest i such that t^(2^i) ≡ 1
        i = 0
        temp = t
        while temp != 1:
            temp = (temp * temp) % p
            i += 1
        
        b = pow(c, 1 << (M - i - 1), p)
        M = i
        c = (b * b) % p
        t = (t * c) % p
        R = (R * b) % p
    
    return R

def matmul(A, B, mod):
    """Matrix multiplication modulo mod."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0] * m for i in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % mod
    return C

def matpow(base, exp, mod):
    """Matrix exponentiation modulo mod."""
    n = len(base)
    res = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
    while exp:
        if exp & 1:
            res = matmul(res, base, mod)
        base = matmul(base, base, mod)
        exp >>= 1
    return res

#═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
#═══════════════════════════════════════════════════════════════════════════════

def getDivisors(n):
    """Returns all divisors of n. Time: O(√n)."""
    divs = set()
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)

def countDivisors(n):
    """Returns number of divisors of n. Time: O(√n)."""
    cnt = 1
    temp = n
    i = 2
    while i * i <= temp:
        exp = 0
        while temp % i == 0:
            exp += 1
            temp //= i
        cnt *= (exp + 1)
        i += 1
    if temp > 1:
        cnt *= 2
    return cnt

def sumDivisors(n):
    """Returns sum of divisors of n. Time: O(√n)."""
    total = 1
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            p = 1
            while temp % i == 0:
                p *= i
                temp //= i
            total *= (p * i - 1) // (i - 1)
        i += 1
    if temp > 1:
        total *= (temp + 1)
    return total

def totient(n):
    """Euler's totient function φ(n). Time: O(√n)."""
    res = n
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            while temp % i == 0:
                temp //= i
            res -= res // i
        i += 1
    if temp > 1:
        res -= res // temp
    return res

def mobius(n):
    """Mobius function μ(n). Time: O(√n)."""
    if n == 1:
        return 1
    prime_count = 0
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            if temp % (i*i) == 0:
                return 0
            prime_count += 1
            while temp % i == 0:
                temp //= i
        i += 1
    if temp > 1:
        prime_count += 1
    return 1 if prime_count % 2 == 0 else -1

def fibonacci(n, mod):
    """Fast Fibonacci using matrix exponentiation. F(0)=0, F(1)=1."""
    if n == 0:
        return 0
    base = [[1, 1], [1, 0]]
    res = matpow(base, n-1, mod)
    return res[0][0]

def buildGCD(A):
    """Precomputes prefix and suffix GCD arrays for range queries."""
    n = len(A)
    prefix = [0] * n
    suffix = [0] * n
    prefix[0] = A[0]
    for i in range(1, n):
        prefix[i] = gcd(prefix[i-1], A[i])
    suffix[n-1] = A[n-1]
    for i in range(n-2, -1, -1):
        suffix[i] = gcd(suffix[i+1], A[i])
    return prefix, suffix

#═══════════════════════════════════════════════════════════════════════════════
# BONUS: COMMON SEQUENCES & FORMULAS
#═══════════════════════════════════════════════════════════════════════════════

def sum_natural(n):
    """Sum of first n natural numbers: n(n+1)/2."""
    return n * (n+1) // 2

def sum_squares(n):
    """Sum of squares of first n natural numbers: n(n+1)(2n+1)/6."""
    return n * (n+1) * (2*n+1) // 6

def sum_cubes(n):
    """Sum of cubes of first n natural numbers: [n(n+1)/2]^2."""
    s = n * (n+1) // 2
    return s * s

def sum_ap(a, d, n):
    """Sum of arithmetic progression: n/2 * (2a + (n-1)d)."""
    return n * (2*a + (n-1)*d) // 2

def sum_gp(a, r, n, mod):
    """Sum of geometric progression: a*(r^n - 1)/(r-1) mod mod. Handles r=1."""
    if r == 1:
        return a * n % mod
    numerator = a * (pow(r, n, mod) - 1) % mod
    denom_inv = pow(r-1, mod-2, mod)
    return numerator * denom_inv % mod