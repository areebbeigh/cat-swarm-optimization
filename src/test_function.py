from math import cos, exp, pi, sqrt

# http://www.geatbx.com/ver_3_3/fcneaso.html
def easom_function(x):
    return -cos(x[0]) * cos(x[1]) * exp(-((x[0] - pi) ** 2) - (x[1] - pi) ** 2)


def bukin_function(x):
    return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)


def ackley_function(x):
    return (
        -exp(-sqrt(0.5 * sum([i**2 for i in x])))
        - exp(0.5 * sum([cos(i) for i in x]))
        + 1
        + exp(1)
    )


def sphere_function(x):
    return sum([i**2 for i in x])
