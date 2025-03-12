"""Generation of rejects by numerical resolution of the inversion equation. Available only for lambda_8."""

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad



def coeffs_8(plaquette):
    """Give the coefficients of the trigonometric function for lambda_8 (cf pdf)

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        (double, double, double, double): a,b,c,d coefficients (cf pdf)
    """
    a = (plaquette[0,0]+plaquette[1,1]).real
    b = (-plaquette[0,0]-plaquette[1,1]).imag
    c = plaquette[2,2].real
    d = plaquette[2,2].imag
    return a,b,c,d

def derivee_f_lambda_8(xi,a,b,c,d):
    """The trigonometric function of the derivative of the energy for lambda_8.

    Args:
        xi (double): angle
        a (double): coefficient of the trig function
        b (double): coefficient of the trig function
        c (double): coefficient of the trig function
        d (double): coefficient of the trig function

    Returns:
        double: derivative of the energy for lambda_8
    """
    return -a/np.sqrt(3)*np.sin(xi/np.sqrt(3))+b/np.sqrt(3)*np.cos(xi/np.sqrt(3))-2*c/np.sqrt(3)*np.sin(2*xi/np.sqrt(3))+2*d/np.sqrt(3)*np.cos(2*xi/np.sqrt(3))

def coeffs_quartic(plaquette):
    """"Return the coefficients of the numerator of the quartic equation for the derivative of the energy, with the dominant coefficient first, obtained using the change of variable t = tan(xi/(2sqrt(3)))."

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list: list of the polynomial coefficients 
    """
    a,b,c,d = coeffs_8(plaquette)
    return [-b+2*d, -2*a+8*c, -12*d, -2*a-8*c, b+2*d]

def real_roots(plaquette):
    """Return the real roots of the numerator of the quartic equation corresponding to the derivative of the energy. The function root calculates the eigenvalues of the companion matrix associated with the polynomial.

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list: list of the real roots
    """
    coeffs_q = coeffs_quartic(plaquette)
    roots = np.roots(coeffs_q)
    real_roots = [r.real for r in roots if np.abs(r.imag) < 1e-6]
    return real_roots

def roots_xi(plaquette):
    """Return the roots of the derivative of the energy for xi ranging from [0, 2pi]

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list: list of roots of the derivative of the energy for lambda_8, variable xi
    """
    real = real_roots(plaquette)
    #print(real)
    a,b,c,d = coeffs_8(plaquette)
    roots_xi = []
    for r in real:
        xi = 2*np.sqrt(3)*np.arctan(r) #Car le chgt de variable est r = tan(xi/2sqrt3))
        
        xi = xi%(2*np.sqrt(3)*np.pi) #Le changement de variable inverse nous donne xi modulo 2sqrt(3)pi
        
        if (derivee_f_lambda_8(xi, a,b,c,d) > 1e-3): #on vérifie qu'on a bien toujours des racines de la dérivée de l'énergie
            print("/!\\ WARNING /!\\ WARNING /!\\ WARNING /!\\")
            print("Evaluation de la racine en xi post chgt de variable : " +str(derivee_f_lambda_8(xi, a,b,c,d)))
            print("/!\\ WARNING /!\\ WARNING /!\\ WARNING /!\\")
            #raise ValueError("Le xi calculé post changement de variable inverse n'est pas une racine de la dérivée de l'énergie")
        if (0 <= xi <= 2*np.pi):
            roots_xi += [xi] #On ne s'intéresse qu'aux racines entre 0 et 2pi
            
    if np.abs(derivee_f_lambda_8(np.sqrt(3)*np.pi,a,b,c,d)) < 1e-6: #on ajoute sqrt(3)pi si c'est racine car ignoré par le chgt de variable (il l'envoie a l'infini)
        roots_xi.append(np.sqrt(3)*np.pi)
    
    roots_xi.sort() #On trie la liste par ordre croissant
    
    return roots_xi

def signe_entre_racines(plaquette):
    """Returns a vector of -1 and 1 depending on the sign of the derivative of the energy between the roots, only for xi between 0 and 2pi (forward crossing).

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list of int: list of 1 or -1 depending on the sign of the function between the roots
    """
    racines = roots_xi(plaquette)
    # Initialisation de la liste des signes
    signes = []
    a,b,c,d = coeffs_8(plaquette)
    if 0 not in racines:
        racines = [0] + racines
    if 2*np.pi not in racines:
        racines += [2*np.pi]
    racines.sort()
    for i in range(len(racines)-1):
        signes += [np.sign(derivee_f_lambda_8((racines[i+1]+racines[i])/2,a,b,c,d))]
    return signes

def intervalle_signes(plaquette):
    """Returns the lists of intervals where the derivative of the energy is positive and negative (useful for the return) on 0 to 2pi

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list of tuple, list of tuples: lists of intervals where the derivative of the energy is positive and negative 
    """
    signes = signe_entre_racines(plaquette)
    racines = roots_xi(plaquette)
    #print(racines)
    intervalles_pos = []  # Liste pour les intervalles positifs
    intervalles_neg = []  # Liste pour les intervalles négatifs

    if 0 not in racines:
        racines = [0] + racines
    if 2*np.pi not in racines:
        racines += [2*np.pi]
    racines.sort()

    for i in range(len(racines)-1):
        if signes[i]==1:
            intervalles_pos += [(racines[i], racines[i+1])]
        if signes[i]==-1:
            intervalles_neg += [(racines[i], racines[i+1])]
    
    return intervalles_pos, intervalles_neg


def derivee_f_lambda_8_plus_aller(t, plaquette):
    """Computes the max between 0 and the derivative of the energy for lambda_8 when xi(t) is moving foward 0 -> 2pi

    Args:
        t (double): angle
        plaquette (numpy.array): 3x3 SU(3) matrix

    Raises:
        ValueError: t must be in [0,2pi]

    Returns:
        double: value of the function
    """
    a,b,c,d = coeffs_8(plaquette)
    if 0<= t <= 2*np.pi:
        return max(0, derivee_f_lambda_8(t,a,b,c,d))
    else:
        raise ValueError("t n'est pas entre 0 et 2pi")

def derivee_f_lambda_8_plus_retour(t,plaquette):
    """Computes the max between 0 and the derivative of the energy for lambda_8 when xi(t) is moving backward 0 <- 2pi 

    Args:
        t (double): backward parametrization of xi (cf pdf xi(t)  = 4pi - t)
        plaquette (numpy.array): 3x3 SU(3) matrix

    Raises:
        ValueError: t must be in [2pi,4pi]

    Returns:
        double: value of the function
    """
    a,b,c,d = coeffs_8(plaquette)
    if 2*np.pi <= t <= 4*np.pi:
        return max(0, -derivee_f_lambda_8(4*np.pi-t,a,b,c,d))
    else:
        raise ValueError("t n'est pas entre 2pi et 4pi")

def derivee_f_lambda_8_plus(t, plaquette):
    """Computes the max between 0 and the derivative of the energy for lambda_8 with parametrization xi(t) and t in 0,4pi

    Args:
        t (double): parameter for xi
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        double: value of the function
    """
    if 0<=t<=2*np.pi:
        return derivee_f_lambda_8_plus_aller(t, plaquette)
    elif 2*np.pi < t <= 4*np.pi:
        return derivee_f_lambda_8_plus_retour(t,plaquette)

def f_8(x, plaquette, a,gamma):
    """The function whose roots are the solutions of the inversion equation for the rejects

    Args:
        x (double): real number
        plaquette (numpy.array): 3x3 SU(3) matrix
        a (double): lower bound of the integral
        gamma (double): real number

    Returns:
        double: integral of max(0, derivative) - gamma
    """
    r,*_= quad(derivee_f_lambda_8_plus, a,x, args=(plaquette))
    return r-gamma

def intervalles_aller_retour(plaquette):
    """Returns the positive intervals for t for forward and return paths

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix

    Returns:
        list of tuples, list of tuples: positive intervals for the foward path, positive intervals for the backwards path (for variable t)
    """
    intervalles_pos, intervalles_neg = intervalle_signes(plaquette)
    intervalles_pos_retour = []
    for i in range(len(intervalles_neg)):
        (a,b) = intervalles_neg[len(intervalles_neg)-i-1] #Les intervalles du retour sont dans l'ordre inverse
        intervalles_pos_retour += [(4*np.pi - b, 4*np.pi-a)] #Le retour commence à 2pi
    return intervalles_pos, intervalles_pos_retour

def contrib(plaquette,a,b):
    """Returns the value of the integral of max(0, E'+) between a and b.

    Args:
        plaquette (numpy.array): 3x3 SU(3) matrix
        a (double): lower bound
        b (double): upper bound

    Raises:
        ValueError: a and b must be in 0,4pi

    Returns:
        double: value of the integral
    """
    if 0<=a<=4*np.pi and 0<=b<=4*np.pi:
        R,*_ = quad(derivee_f_lambda_8_plus, a,b, args=(plaquette))
        return R
    else:
        raise ValueError("Mauvaises bornes pour l'intervalle d'intégration")

def reduce_gamma_8(gamma, plaquette):
    """Takes a gamma, reduces it to its value in the contribution of the interval containing the solution, and determines the interval in which to solve the equation.

    Args:
        gamma (double): real number
        plaquette (numpy.array): 3x3 SU(3) matrix

    Raises:
        ValueError: if the computed bounds of the interval are in the wrong order

    Returns:
        double, double, double: reduced gamma, lower bound, upper bound
    """
    contrib_aller = contrib(plaquette,0,2*np.pi)
    contrib_retour = contrib(plaquette, 2*np.pi, 4*np.pi)
    contrib_total = contrib_aller+contrib_retour

    
    gamma_r = gamma%contrib_total
    int_aller, int_retour = intervalles_aller_retour(plaquette)
    a,b = 0,0 #Les  bornes de l'intervalle qu'on va renvoyer
    if gamma_r <= contrib_aller:
        for i in range(len(int_aller)):
            a_i, b_i = int_aller[i]
            if (gamma_r <= contrib(plaquette,a_i,b_i)):
                a = a_i
                b = b_i
                break
            else:
                gamma_r -= contrib(plaquette,a_i,b_i)
    else:
        gamma_r -= contrib_aller
        for i in range(len(int_retour)):
            a_i, b_i = int_retour[i]
            if (gamma_r <= contrib(plaquette,a_i,b_i)):
                a = a_i
                b = b_i
                break
            else:
                gamma_r -= contrib(plaquette,a_i,b_i)
    if b<a:
        raise ValueError("Intervalle dans le mauvais sens")
    
    return gamma_r, a,b

def solve_8(gamma, plaquette):
    """Solves the inversion equation for lambda_8 for a given number on the rhs of the inversion equation and a given plaquette

    Args:
        gamma (double): the rhs of the inversion equation
        plaquette (numpy.array): 3x3 SU(3) matrix

    Raises:
        ValueError: the numerically computed reject angle is not right (we don't retrieve gamma)

    Returns:
        double: solution of the inversion equation
    """
    
    gamma_r, a,b= reduce_gamma_8(gamma,plaquette)
    
    #try:
    solution = brentq(f_8, a, b, args = (plaquette,a,gamma_r))
    
    #On vérifie que la solution calculée nous donne bien le gamma réduit
    #On coupe l'intégrale en 2 pour éviter la discontinuité en 2pi qui met un message d'avertissement
    contrib_aller = contrib(plaquette,0,2*np.pi)
    contrib_retour = contrib(plaquette, 2*np.pi, 4*np.pi)
    contrib_total = contrib_aller+contrib_retour
    
    rsol = 0
    if solution < 2*np.pi :
        rsol,*_=quad(derivee_f_lambda_8_plus, 0, solution, args=(plaquette))
    else:
        rsol,*_=quad(derivee_f_lambda_8_plus, 2*np.pi, solution, args=(plaquette)) #on coupe l'intégrale en 2 pour éviter la discontinuité en 2pi qui met un message d'avertissement
        rsol = rsol +contrib_aller
    
    gamma_check = gamma%contrib_total
    if (gamma_check - rsol > 1e-1):
        print(gamma_check-rsol)
        raise ValueError("Le rejet calculé ne correspond pas au gamma")
    #si on est ici, c'est que notre solution convient. Il faut maintenant donner l'angle correspondant : si solution est entre 0 et 2pi,
    #c'est bon, sinon on rend 4pi - sol
    if solution > 2*np.pi:
        solution = 4*np.pi - solution
    return solution