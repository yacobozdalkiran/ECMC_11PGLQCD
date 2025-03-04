"""Generation of rejects by analytical resolution of the inversion equation. Available only for lambda_2, lambda_3 and lambda_5."""

import math
import numpy as np

def solve_trig_eq(a, b, c, gamma):
    """Solves the equation acos(x)+bsin(x)+c = gamma modulo 2pi.

    Args:
        a (double): coefficient of cos
        b (double): coefficient of sin
        c (double): constant coefficient 
        gamma (double): rhs of the equation

    Raises:
        ValueError: if gamma-c must be > sqrt(a^2 + b^2), no solution possible

    Returns:
        double: the 2 solutions ordered
    """
    # Vérification de la condition c / sqrt(a^2 + b^2) ∈ [-1, 1]
    R = math.sqrt(a**2 + b**2)
    rhs = gamma-c
    if abs(rhs) > R:
        raise ValueError("No solution: |gamma-c| must be ≤ sqrt(a^2 + b^2).")
    
    # Calcul de theta
    theta = math.atan2(b, a)  # atan2 donne le bon quadrant pour cos(theta) et sin(theta)
    
    # Calcul des solutions principales
    cos_argument = rhs / R
    acos_value = math.acos(cos_argument)
    
    # Deux solutions principales
    x1 = theta + acos_value
    x2 = theta - acos_value
    
    # Réduction dans [0, 2*pi]
    x1 = x1 % (2 * math.pi)
    x2 = x2 % (2 * math.pi)
    
    return min(x1,x2), max(x1,x2)

def coeffs(plaquette, index_lambda):
    """Computes the a,b,c coefficients for a given plaquette and gell-mann index

    Args:
        plaquette (numpy.array): plaquette
        index_lambda (int): gell-man index

    Raises:
        ValueError: the gell mann index must be 2,3 or 5

    Returns:
        double,double,double: the a,b,c coefficients
    """
    a,b,c = 0,0,0
    if index_lambda not in [2,3,5]:
        raise ValueError("index_lambda invalide")
    
    if (index_lambda==2):
        a = (plaquette[0,0]+plaquette[1,1]).real
        b = (plaquette[1,0]-plaquette[0,1]).real
        c = plaquette[2,2].real
    if (index_lambda==3):
        a = (plaquette[0,0]+plaquette[1,1]).real
        b = (-plaquette[0,0]+plaquette[1,1]).imag
        c = plaquette[2,2].real
    if (index_lambda==5):
        a = (plaquette[0,0]+plaquette[2,2]).real
        b = (plaquette[2,0]-plaquette[0,2]).real
        c = plaquette[1,1].real
    return a,b,c

def signes(a,b):
    """Returns a tuple of either 1, -1, or 0 based on the signs of the real numbers a and b

    Args:  
    a (double): real number  
    b (double): real number  

    Returns:  
        (int, int): returns (1, _) if a > 0, (-1, _) if a < 0, (0, _) if a = 0; same for b  
    """ 
    if a==0 and b==0:
        return (0,0)
    elif (a==0):
        return (0, np.abs(b)/b)
    elif b == 0:
        return (np.abs(a)/a, 0)
    else:
        return (np.abs(a)/a, np.abs(b)/b)

def reduce(gamma,plaquette,index_lambda):
    """Returns gamma with the periodic contributions corresponding to index_lambda removed, see Tables 3 and 4 of the PDF document.

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix, defines the contribution to remove

    Raises:
        ValueError: index_lambda must be in 2,3,5

    Returns:
        double: reduced gamma
    """
    a,b,*_= coeffs(plaquette, index_lambda)
    if (index_lambda==2) or (index_lambda==5): #dans ce cas, l'angle est entre 0 et pi/2 -> table 3
        if (a*b >= 0):
            R = 2*np.sqrt(a**2 + b**2) - np.abs(a) - np.abs(b)
            mod = gamma%R
            #if mod > R/2:
            #    mod -= R
            
        else:
            R = np.abs(a)+np.abs(b)
            mod = gamma%R
            #if mod > R/2:
            #    mod -= R
            
    elif (index_lambda==3): #dans ce cas, l'angle est entre 0 et pi -> table 4
        R = 2*np.sqrt(a**2+b**2)
        mod = gamma%R
        #if mod > R/2:
        #    mod -= R
    else:
        raise ValueError("Wrong value for index_lambda")
    return mod

def gamma_aller_retour(gamma, plaquette, index_lambda):
    """Takes a gamma, reduces it, and returns True if it is reached during the forward path, False otherwise.

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix

    Returns:
        bool: True if gamma is reached during the forward path, False otherwise
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b,*_ = coeffs(plaquette, index_lambda)
    sign = signes(a,b)
    aller = 0
    match index_lambda:
        case 2|5:
            match sign:
                case (1,1):
                    aller = np.sqrt(a**2+b**2)-a
                case (-1,-1):
                    aller = np.sqrt(a**2+b**2)+b
                case (1,-1):
                    aller = 0
                case (-1,1):
                    aller = np.abs(a)+np.abs(b)
                case (1,0):
                    aller = 0
                case(-1,0):
                    aller = np.abs(a)
                case (0,1):
                    aller = np.abs(b)
                case (0,-1):
                    aller = 0
        case 3:
            match sign:
                case (1,0):
                    aller = 0
                case (-1,0):
                    aller = -2*a
                case (0,1):
                    aller = b
                case (0,-1):
                    aller = -b
                case _:
                    aller = np.sqrt(a**2+b**2)-a
    return (gamma <= aller)

def retire_contrib_aller(gamma, plaquette, index_lambda):
    """Removes a forward contribution from gamma.

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix, defines the contribution to remove

    Returns:
        double: reduced gamma
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b,*_ = coeffs(plaquette, index_lambda)
    sign = signes(a,b)
    aller = 0
    match index_lambda:
        case 2|5:
            match sign:
                case (1,1):
                    aller = np.sqrt(a**2+b**2)-a
                case (-1,-1):
                    aller = np.sqrt(a**2+b**2)+b
                case (1,-1):
                    aller = 0
                case (-1,1):
                    aller = np.abs(a)+np.abs(b)
        case 3:
            aller = np.sqrt(a**2+b**2)-a
    
    return gamma - aller

def intervalle(gamma, plaquette, index_lambda):
    """Takes a gamma, reduces it, checks if it is reached during a forward or backward path, and returns the interval in which to search for the rejection angle, given that the variable t represents the total displacement (cf pdf for xi parametrization)

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix

    Returns:
        (double,double): the bound of the intervall on which gamma is attained for the t variable (cf pdf for xi(t) parametrization)
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b,*_ = coeffs(plaquette, index_lambda)
    sign = signes(a,b)
    a_r = gamma_aller_retour(gamma, plaquette, index_lambda)
    bound1, bound2 = 0,0
    if a_r:
        match index_lambda:
            case 2|5:
                match sign:
                    case (1,1):
                        bound1, bound2 = 0, np.arctan(b/a)
                    case (-1,-1):
                        bound1, bound2 = np.arctan(b/a), np.pi/2
                    case (1,-1):
                        bound1, bound2 = 0, 0
                    case (-1,1):
                        bound1, bound2 = 0, np.pi/2
            case 3:
                match sign:
                    case (1,1):
                        bound1, bound2 = 0, np.arctan(b/a)
                    case (-1,-1):
                        bound1, bound2 = np.arctan(b/a), np.pi
                    case (1,-1):
                        bound1, bound2 = np.arctan(b/a)+np.pi, np.pi
                    case (-1,1):
                        bound1, bound2 = 0, np.arctan(b/a)+np.pi
    else:
        match index_lambda:
            case 2|5:
                match sign:
                    case (1,1):
                        bound1, bound2 = np.pi/2, np.pi-np.arctan(b/a)
                    case (-1,-1):
                        bound1, bound2 = np.pi-np.arctan(b/a), np.pi
                    case (1,-1):
                        bound1, bound2 = np.pi/2, np.pi
                    case (-1,1):
                        bound1, bound2 = 0,0
            case 3:
                match sign:
                    case (1,1):
                        bound1, bound2 = np.pi, 2*np.pi-np.arctan(b/a)
                    case (-1,-1):
                        bound1, bound2 = 2*np.pi-np.arctan(b/a), 2*np.pi
                    case (1,-1):
                        bound1, bound2 = np.pi-np.arctan(b/a), 2*np.pi
                    case (-1,1):
                        bound1, bound2 = np.pi, np.pi-np.arctan(b/a)
    return bound1,bound2

def coeffs_ana(gamma, plaquette, index_lambda):
    """Takes a gamma, reduces it, checks if it is forward or backwards; if backwards, returns a and b correctly modified according to index_lambda.

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix
    Returns:
        double, double: +/-a,+/-b according to the case 
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b,*_ = coeffs(plaquette, index_lambda)
    sign = signes(a,b)
    a_r = gamma_aller_retour(gamma, plaquette, index_lambda)
    new_a, new_b = a,b

    if (not a_r) and not((a==0)or(b==0)):
        match index_lambda:
            case 2|5:
                new_a = -a
            case 3:
                new_b = -b
    return new_a, new_b


def coeff_c(gamma, plaquette, index_lambda):
    """Returns the value of c' (cf pdf) needed to solve the reject equation

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix

    Raises:
        ValueError: if the corresponding (a,b) sign does not match the assessement of foward/backward path (cf tables)
        ValueError: if the corresponding (a,b) sign does not match the assessement of foward/backward path (cf tables)

    Returns:
        double: the c' coefficient
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b,*_ = coeffs(plaquette, index_lambda)
    sign = signes(a,b)
    a_r = gamma_aller_retour(gamma, plaquette, index_lambda)
    c = 0
    r = np.sqrt(a**2+b**2)
    if a_r:
        match index_lambda:
            case 2|5:
                match sign:
                    case (1,1):
                        c = -a
                    case (-1,-1):
                        c = r
                    case (1,-1):
                        raise ValueError("On ne peut pas être dans un aller")
                    case (-1,1):
                        c = -a
            case 3:
                match sign:
                    case (1,1):
                        c = -a
                    case (-1,-1):
                        c = r
                    case (1,-1):
                        c = r
                    case (-1,1):
                        c = -a
    else:
        match index_lambda:
            case 2|5:
                match sign:
                    case (1,1):
                        c = -b
                    case (-1,-1):
                        c = r
                    case (1,-1):
                        c = -b
                    case (-1,1):
                        raise ValueError("On ne peut pas être dans un retour")
            case 3:
                match sign:
                    case (1,1):
                        c = a
                    case (-1,-1):
                        c = r
                    case (1,-1):
                        c = r
                    case (-1,1):
                        c = a
    return c

def rejet_ana(gamma, plaquette, index_lambda):
    """Takes a gamma, reduces it, checks if it is forward or backward; if backward, removes the forward contribution, then solves the inversion equation in the correct interval and returns the corresponding angle xi(t).

    Args:
        gamma (double): the real number to reduce
        plaquette (numpy.array): the plaquette used to compute a,b coefficients
        index_lambda (int): the index of the gell-mann matrix

    Raises:
        ValueError: if we have 2 solutions in the determined interval
        ValueError: if we have no solutions in the determined interval

    Returns:
        double: the reject angle xi(t)
    """
    gamma = reduce(gamma, plaquette, index_lambda)
    a,b = coeffs_ana(gamma, plaquette, index_lambda)
    sign = signes(a,b)
    a_r = gamma_aller_retour(gamma, plaquette, index_lambda)

    if a==0 or b==0:
        
        match index_lambda:
            case 2|5:
                match sign:
                    case (1,0):
                        return np.pi - np.acos(gamma/(-a))
                    case (-1,0):
                        return np.acos(gamma/a +1)
                    case (0,1):
                        return np.asin(gamma/b)
                    case (0,-1):
                        return np.pi - np.asin(1+gamma/b) 
            case 3:
                match sign:
                    case (1,0):
                        return 2*np.pi - np.acos(gamma/a - 1)
                    case (-1,0):
                        return np.acos(gamma/a+1)
                    case (0,1):
                        return np.asin(gamma/b)
                    case (0,-1):
                        return np.asin(gamma/b+1)

    else:
        bound1, bound2 = intervalle(gamma, plaquette, index_lambda)
        c = coeff_c(gamma, plaquette, index_lambda)
        x1,x2 = 0,0
        if a_r:
            x1, x2 = solve_trig_eq(a,b,c,gamma)
        else:
            gamma = retire_contrib_aller(gamma, plaquette, index_lambda)
            x1, x2 = solve_trig_eq(a,b,c,gamma)
        sol1,sol2 = False, False
        sol = 0
        if (bound1 <= x1 <= bound2):
            sol1 = True
            sol = x1
        if (bound1 <= x2 <= bound2):
            sol2 = True
            sol = x2
        if (sol1==True) and (sol2==True):
            print(gamma, plaquette, index_lambda)
            raise ValueError("Deux solutions possibles dans l'intervalle")
        elif (sol1==False) and (sol2==False):
            print(gamma, plaquette, index_lambda)
            raise ValueError("Aucune solution possible dans l'intervalle")
        #sol est donc le t rejet. On veut maintenant le xi rejet
        xi = 0
        if a_r:
            xi = sol
        else:
            if index_lambda==2 or index_lambda==5:
                xi = np.pi - sol
            if index_lambda==3:
                xi = 2*np.pi - sol
        return xi
    