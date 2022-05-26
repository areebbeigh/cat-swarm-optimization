from math import cos, exp, pi

# http://www.geatbx.com/ver_3_3/fcneaso.html
def easom_function(x):
    return -cos(x[0]) * cos(x[1]) * exp(-((x[0] - pi) ** 2) - (x[1] - pi) ** 2)
