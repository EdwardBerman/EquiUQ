import mpmath as mp

# Parameters
mu = 0.5
sigma = 0.1
a = 0
b = 1

# Standard normal pdf and cdf
phi = lambda x: mp.exp(-0.5*x**2) / mp.sqrt(2*mp.pi)
Phi = lambda x: 0.5 * (1 + mp.erf(x/mp.sqrt(2)))

# Normalization constant for truncated normal
alpha = (a - mu) / sigma
beta = (b - mu) / sigma
Z = Phi(beta) - Phi(alpha)

# Truncated normal pdf
def truncated_normal_pdf(x):
    return (1/sigma) * phi((x - mu)/sigma) / Z

# Integrand for |0.5 - x|
integrand = lambda x: abs(0.5 - x) * truncated_normal_pdf(x)

# Numerical integration
result = mp.quad(integrand, [a, b])
bound = result + 0.5
print(bound)

