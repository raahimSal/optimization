def constraint(z):
    """ Constraint outlined in (13) in the paper. It is returned like 0 - c, due to the how the minimizers in scipy work. The 0-c is intereted as 0 >= c, which simplifies to (13)"""
    c = 3*z[2]**2-2*z[1]
    return 0 - c