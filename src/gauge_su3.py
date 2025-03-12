"""Useful functions to create/manipulate gauge configurations and SU(3) matrices"""

import copy
import numpy as np
from random import random

def el_2(xi):
    """Generates exp(xi*lambda_2) where lambda_2 is the 2nd Gell-Man matrice

    Args:
        xi (double): angle

    Returns:
        numpy.array: exp(xi*lambda_2)
    """
    return np.array([[np.cos(xi), np.sin(xi), 0],
                     [-np.sin(xi), np.cos(xi), 0],
                     [0,0,1]])


def el_3(xi):
    """Generates exp(xi*lambda_3) where lambda_3 is the 3rd Gell-Man matrice

    Args:
        xi (double): angle

    Returns:
        numpy.array: exp(xi*lambda_3)
    """
    return np.array([[np.exp(xi*1j), 0, 0],
                     [0,np.exp(-xi*1j),0],
                     [0, 0,1]])


def el_5(xi):
    """Generates exp(xi*lambda_5) where lambda_5 is the 5th Gell-Man matrice

    Args:
        xi (double): angle

    Returns:
        numpy.array: exp(xi*lambda_5)
    """
    return np.array([[np.cos(xi), 0, np.sin(xi)],
                     [0,1,0],
                     [-np.sin(xi), 0,np.cos(xi)]])

def el_8(xi):
    """Generates exp(xi*lambda_8) where lambda_8 is the 8th Gell-Man matrice

    Args:
        xi (double): angle

    Returns:
        numpy.array: exp(xi*lambda_8)
    """
    return np.array([[np.exp((xi/np.sqrt(3))*1j), 0, 0],
                     [0,np.exp((xi/np.sqrt(3))*1j),0],
                     [0, 0,np.exp(-2*(xi/np.sqrt(3))*1j)]])
def el(xi,index):
    """""Generates the exp(xi*lambda_i) where lambda_i is the i_th Gell-Mann matrice

    Args:
        xi (double): angle
        index (int): index of the Gell-Mann matrice, can only be 2,3,5 or 8

    Raises:
        ValueError: if index is not 2,3,5 or 8

    Returns:
        numpy.array: exp(xi*lambda_i)
    """""
    if (index not in [2,3,5,8]):
        raise ValueError("index must be 2,3,5 or 8")
    if (index == 2):
        return el_2(xi)
    if (index == 3):
        return el_3(xi)
    if (index == 5):
        return el_5(xi)
    if (index == 8):
        return el_8(xi)

def mat_rand_su3():
    """Generates a random SU(3) matrice using the parametrization of https://arxiv.org/abs/physics/9708015

    Returns:
        numpy.array: random SU(3) matrix
    """
    alpha, gamma, a, c = random()*np.pi,random()*np.pi,random()*np.pi,random()*np.pi
    ybeta, yb, ytheta = random(),random(),random()
    beta = np.arccos(1-2*ybeta)/2
    b = np.arccos(1-2*yb)/2
    theta = np.arcsin(ytheta**(1/4))
    phi = random()*2*np.pi
    return el_3(alpha) @ el_2(beta) @ el_3(gamma) @ el_5(theta) @ el_3(a) @ el_2(b) @ el_3(c) @ el_8(phi)

def est_SU3(U, tol=1e-10):
    """Checks wether a matrix is in SU(3)

    Args:
        U (numpy.array): 3x3 matrix to check
        tol (double, optional): Tolerance of numerical checks. Defaults to 1e-10.

    Returns:
        bool: True is the matrix is in SU(3), False if not
    """
    # Vérifier l'unitarité : U * U_dagger = I
    U_dagger = np.conjugate(U.T)
    identite = np.eye(3, dtype=complex)
    
    if not np.allclose(U @ U_dagger, identite, atol=tol):
        return False  # La matrice n'est pas unitaire
    
    # Vérifier que le déterminant est égal à 1 : det(U) = 1
    if not np.isclose(np.linalg.det(U), 1, atol=tol):
        return False  # Le déterminant n'est pas égal à 1
    
    return True  # La matrice est dans SU(3)

def init_conf(L=3, T=3, cold=True):
    """Returns a 1+1d gauge configuration of size L*T

    Args:
        L (int, optional): Spatial dimension of the lattice. Defaults to 3.
        T (int, optional): Temporal dimension of the lattice. Defaults to 3.
        cold (bool, optional): If True, all matrix initialized to identity, else all matrices are random. Defaults to True.

    Returns:
        numpy.array: L*T 1+1d lattice gauge configuration
    """
    mu = 2
    conf = np.zeros((L,T,mu,3,3), dtype=complex)
    for x in range(L):
        for t in range(T):
            for u in range(mu):
                if (cold):
                    conf[x,t,u] = np.eye(3)
                else:
                    conf[x,t,u] = mat_rand_su3()
    return conf

def conf_est_SU3(conf, tol=1e-10):
    """Checks wheter all the matrices in a lattice gauge configuration are SU(3) matrices

    Args:
        conf (numpy.array): The gauge configuration to check
        tol (double, optional): Tolerance of numerical checks. Defaults to 1e-10.

    Returns:
        bool: True if all the matrices in a lattice gauge configuration are SU(3) matrices, if not False
    """
    L,T,mu,*_ = conf.shape
    res = True
    for i in range(L):
        for j in range(T):
            for u in range(mu):
                if not est_SU3(conf[i,j,u]):
                    res = False
    return res

def get_link(conf, x, t, mu):
    """Gets a link in a gauge configuration with periodic boundary conditions

    Args:
        conf (numpy.array): gauge configuration
        x (int): space coordinate of the link
        t (int): time coordinate of the link
        mu (int): lorentz coordinate of the link

    Returns:
        numpy.array: the matrix of the link
    """
    L,T,*_=conf.shape
    x_pbc = x % L  # Appliquer périodicité en x
    t_pbc = t % T  # Appliquer périodicité en t
    return conf[x_pbc, t_pbc, mu]

def update_link(conf, x, t, mu, new_matrix):
    """Updates a gauge configuration by multiplying the selected link with a matrix

    Args:
        conf (numpy.array): gauge configuration
        x (int): space coordinate of the link
        t (int): time coordinate of the link
        mu (int): lorentz coordinate of the link
        new_matrix (numpy.array): the updating matrix
    """
    L,T,*_=conf.shape
    x_pbc = x % L
    t_pbc = t % T
    conf[x_pbc, t_pbc, mu] = new_matrix@conf[x_pbc, t_pbc, mu]

def calculate_plaquette(gauge_config, x, t):
    """Computes the plaquette at a given lattice site (x,t,mu). The operation is : U(x,t,0) U(x+1,t,1) U*(x,t+1,0)^T U*(x,t,1)^T

    Args:
        gauge_config (numpy.array): gauge configuration
        x (int): space coordinate of the link
        t (int): time coordinate of the link

    Returns:
        numpy.array: the plaquette (3x3 matrix)
    """
    L,T,*_=gauge_config.shape
    # Liens nécessaires
    Ux = get_link(gauge_config, x%L, t%T, 0)  # Lien dans x
    Ut = get_link(gauge_config, x%L, t%T, 1)  # Lien dans t
    Ux_t = get_link(gauge_config, x%L, (t + 1) % T, 0)  # Lien dans x au site (x, t+1)
    Ut_x = get_link(gauge_config, (x + 1) % L, t%T, 1)  # Lien dans t au site (x+1, t)
    
    # Plaquette
    plaquette = Ux @ Ut_x @ np.conj(Ux_t.T) @ np.conj(Ut.T)
    return plaquette

def calculate_action(conf, beta):
    """Computes the Wilson action of a gauge configuration

    Args:
        conf (numpy.array): gauge configuration
        beta (double): inverse coupling constant of the Wilson action

    Returns:
        double: the Wilson action of the gauge configuration
    """
    L,T,*_=conf.shape
    action = 0
    for x in range(L):
        for t in range(T):
            plaquette_xt = calculate_plaquette(conf,x,t)
            i = np.eye(3,dtype=complex)
            action += np.trace(i-plaquette_xt).real
    action = action*(beta/3)
    action = action/ (L*T) #Normalisation
    return action

def transf_gauge(conf):
    """Returns a copy of the gauge configuration where all the links were multiplied by a random SU(3) matrix to simulate a random local gauge transform.

    Args:
        conf (numpy.array): gauge configuration

    Returns:
        numpy.array: the transformed configuration
    """
    L,T,*_ = conf.shape
    gauge_transf = np.zeros((L,T,3,3), dtype = complex)
    conf_transf = copy.deepcopy(conf)
    for x in range(L):
        for t in range(T):
            gauge_transf[x,t] = mat_rand_su3()
    for x in range(L):
        for t in range(T):
            for mu in range(2):
                conf_transf[x,t,mu] = gauge_transf[x,t] @ conf_transf[x,t,mu] @ np.conjugate(gauge_transf[(x + 1 - mu)%L, (t + mu)%T]).T
    return conf_transf

def calculate_diff_action(conf, x,t,mu,m, beta):
    """Computes delta S = new action - old action the action difference where the x,t,mu link is multiplied by m

    Args:
        conf (numpy.array): gauge configuration
        x (int): space coordinate of the link
        t (int): time coordinate of the link
        mu (int): lorentz coordinate of the link
        m (numpy.array): updating SU(3) matrix
        beta (double): Wilson action inverse coupling constant

    Returns:
        double: the difference of action
    """
    conf_new = conf
    update_link(conf_new,x,t,mu,m)
    action_old = calculate_action(conf, beta)
    action_new = calculate_action(conf_new,beta)
    diff = action_new-action_old
    return diff