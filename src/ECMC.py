from analytical_reject import *
from gauge_su3 import *
from numerical_reject import *
import sys
from IPython.display import clear_output
from random import randrange

def ECMC_step(conf, x, t, mu, index_lambda, beta):
    """"Performs an ECMC step: returns the matrix X = exp(i xi lambda_{indice_lambda}) and the angle xi of the rejection event for the link (x,t,mu) of the configuration conf, as well as next_link, the link in the plaquette responsible for the rejection that will need to be changed afterward. index_lambda can only take values 2, 3, 5, or 8.

    Args:
        conf (numpy.array): 1+1d lattice gauge configuration
        x (int): spatial coordinate of the link
        t (int): temporal coordinate of the link
        mu (int): lorentz coordinate of the link
        index_lambda (int): direction of updating
        beta (double): Wilson action inverse coupling constant

    Raises:
        ValueError: index_lambda must be 2,3,5 or 8

    Returns:
        numpy.array, double, (int,int,int): updating matrix X, angle of updating xi, next link to update (x,t,mu)
    """
    if (index_lambda!=2) and (index_lambda!=3) and (index_lambda!=5) and (index_lambda!=8):
        raise ValueError("Wrong index_lambda value, must be in {2,3,5,8}")
    #On calcule les deux plaquettes contenant le lien :
    L,T,*_ = conf.shape
    plaquette_1 = calculate_plaquette(conf,x,t)
    plaquette_2 = np.eye(3, dtype = complex)
    if (mu==0):
        plaquette_2 = calculate_plaquette(conf, x, t-1)
    if (mu==1):
        plaquette_2 = calculate_plaquette(conf, x-1, t) 

    #Le xi rejet qui sera rendu à la fin
    xi = 0
    
    #xi rejet dans les plaquettes 1 et 2
    xi_1 = 0
    xi_2 = 0
    
    p_1 = random()
    p_2 = random()
    gamma_1 = (-3/beta)*math.log(p_1)
    gamma_2 = (-3/beta)*math.log(p_2)

    
    match index_lambda:
        case 8:
            try:
                xi_1 = solve_8(gamma_1, plaquette_1)
                xi_2 = solve_8(gamma_2, plaquette_2)
            except Exception as e:  # Capture toutes les exceptions standard
                print(gamma_1,plaquette_1)
                print(gamma_2,plaquette_2)
                print(f"Une erreur est survenue : {e}")
                raise
        case _:
            try:
                xi_1 = rejet_ana(gamma_1, plaquette_1, index_lambda)
                xi_2 = rejet_ana(gamma_2, plaquette_2, index_lambda)
            except Exception as e:  # Capture toutes les exceptions standard
                print(gamma_1,plaquette_1,index_lambda)
                print(gamma_2,plaquette_2,index_lambda)
                print(f"Une erreur est survenue : {e}")
                raise
    #On choisit le xi le plus petit
    xi = min(xi_1, xi_2)

    next_link=(-1,-1,-1)
    
    #On choisit le prochain lien a modifier dans la plaquette responsable du refus, en prenant le prochain lien dans le sens trigo
    #(les plaquettes sont orientées dans le sens trigo)
    if (xi_1<=xi_2):
        #print("Refus dans la 1ère plaquette")
        if (mu==0):
            next_link = ((x+1)%L,t,1)
        if (mu==1):
            next_link = (x,t,0)
        
    else:
        #print("Refus dans la 2ème plaquette")
        if (mu==0):
            next_link = (x,(t-1)%T,1)
        if (mu==1):
            next_link = ((x-1)%L,(t-1)%T,0)

    X = el(xi, index_lambda)

    return X, xi, next_link


def ECMC_step_l(conf, x, t, mu, beta, angle_l, param_lambda_l, param_pos_l):
    """Returns the configuration conf after ECMC steps starting from the point (x, t, mu) until the total angle traveled reaches angle_l.

    Args:
        conf (numpy.array): 1+1d lattice gauge configuration
        x (int): spatial coordinate of the link
        t (int): temporal coordinate of the link
        mu (int): lorentz coordinate of the link
        index_lambda (int): direction of updating
        beta (double): Wilson action inverse coupling constant
        angle_l (double): total distance upon which we select a sample
        param_lambda_l (double): poisson law parameter of the total distance upon which we change the direction of updating
        param_pos_l (double): poisson law parameter of the total distance upon which we randomly change the updated link

    Raises:
        ValueError: angle_l must be positive
    """
    if (angle_l<=0):
        raise ValueError("angle_l doit être strictement positif")
        
    L, T, *_ = conf.shape
    
    #Position du lien courant (celui qui se fait modifier)
    x_c = x%L
    t_c = t%T
    mu_c = mu

    #génération des longueurs de déplacement au bout desquelles on change la position du lien de jauge à modifier et la direction de déplacement
    pos_l = np.random.exponential(scale=param_pos_l)
    lambda_l = np.random.exponential(scale=param_lambda_l)
    print("pos_l = " +str(pos_l))
    print("lambda_l = "+str(lambda_l))
    #angle total
    angle = 0
    
    
    #liste des indices de lambda dans l'ordre et déplacement total depuis le dernier changement de direction
    index_lambda = [8, 3, 2, 3, 5, 3, 2, 3]
    i = 0
    angle_lambda = 0

    #déplacement total depuis le dernier changement de lien aléatoire
    angle_pos = 0

    #Pour afficher la progression
    printer = 0
    counter_rejects = 0
    print("*--------------------------------------------------------------------* ")
    #prefixe.append("*--------------------------------------------------------------------* ")
    while (angle < angle_l):
        
        X, xi, next_link = ECMC_step(conf, x_c, t_c, mu_c, index_lambda[i], beta)
        
        if (angle + xi  > angle_l) or (angle_pos + xi > pos_l) or (angle_lambda + xi > lambda_l):
            
            diff_angle = angle_l - angle
            diff_pos = pos_l - angle_pos
            diff_lambda = lambda_l - angle_lambda
            min_diff = min(diff_angle, diff_pos, diff_lambda)

            X = el(min_diff, index_lambda[i])
            update_link(conf, x_c, t_c, mu_c, X)
            
            if (min_diff == diff_angle):
                angle = angle_l
                print("Angle limite atteint, " + str(counter_rejects) +" rejets générés")
                
            if (min_diff == diff_pos):
                x_c, t_c, mu_c = randrange(0, L), randrange(0,T), randrange(0,2)
                angle_pos = 0
                pos_l = np.random.exponential(scale=param_pos_l)
                print("Changement de lien aléatoire")
                print("New pos_l = " +str(pos_l) +"\n")
                
            if (min_diff == diff_lambda):
                x_c, t_c, mu_c = next_link
                i += 1
                i = i%len(index_lambda)
                angle_lambda = 0
                lambda_l = np.random.exponential(scale=param_lambda_l)
                print(r"Changement de direction de déplacement, on passe à $\lambda_" +str(index_lambda[i])+r"$")
                print("New lambda_l = " +str(lambda_l)+"\n")
        else :
            update_link(conf, x_c, t_c, mu_c, X)
            x_c, t_c, mu_c = next_link
            angle += xi
            angle_pos += xi
            angle_lambda += xi
            
        counter_rejects+=1    
        #Module pour afficher la progression
        #clear_output(wait=True)
        #for s in prefixe:
        #    print(s)
        #print(f"Distance totale parcourue : {angle/angle_l*100:.2f}% avant génération de sample, {counter_rejects} rejets générés")
        #print(f"{angle_pos/pos_l *100:.2f}% avant changement de lien, distance limite : {pos_l}")
        #print(f"{angle_lambda/lambda_l *100:.2f}% avant changement de direction, distance limite : {lambda_l}")
        #print("*--------------------------------------------------------------------*")
        #Fin du module pour afficher la progression
        
        if (angle>printer*(angle_l/10)):
            print(str(printer * 10) + "% de angle_l parcourus, " +str(counter_rejects) + " rejets générés")
            printer += 1
        
    
    print("Sample généré !")

def ECMC_samples(L=4, T=4, cold = True, beta=2.55, angle_l=120, param_lambda_l=15, param_pos_l=30, n=5):
    """Returns n samples of gauge configurations sampled from the probability distribution induced by the Wilson action

    Args:
        L (int, optional): spatial length of the lattice. Defaults to 4.
        T (int, optional): temporal length of the lattice. Defaults to 4.
        cold (bool, optional): True if cold start, False for hot start. Defaults to True.
        beta (double): Wilson action inverse coupling constant
        angle_l (double): total distance upon which we select a sample
        param_lambda_l (double): poisson law parameter of the total distance upon which we change the direction of updating
        param_pos_l (double): poisson law parameter of the total distance upon which we randomly change the updated link
        n (int, optional): number of samples to generate. Defaults to 5.

    Returns:
        list of numpy.array: list of n samples of gauge configurations
    """
    conf = init_conf(L, T, cold)
    x, t, mu = randrange(0,L), randrange(0,T), randrange(0,2)
    sample = [copy.deepcopy(conf)]
    for i in range(n):
        print("Génération du sample " +  str(i+1) + "...")
        print("-------------------------------------------------------------------")
        #prefixe.append("Génération du sample " +  str(i+1) + "...")
        ECMC_step_l(conf, x, t, mu, beta, angle_l, param_lambda_l, param_pos_l)
        conf_sample = copy.deepcopy(conf)
        sample += [conf_sample]
        #prefixe = prefixe[(i+2):]
        print("Sample " + str(i+1) + " généré !")
        #prefixe.append("Sample " + str(i+1) + " généré !")
    print("Tous les samples ont été générés avec succès !")
    return sample

def walker_action(L = 4, T = 4, cold = True, beta = 2.55, angle_l = 120, param_lambda_l = 15, param_pos_l = 30, n=5):
    """Returns the list of actions of n samples of gauge configurations sampled from the probability distribution induced by the Wilson action

    Args:
        L (int, optional): spatial length of the lattice. Defaults to 4.
        T (int, optional): temporal length of the lattice. Defaults to 4.
        cold (bool, optional): True if cold start, False for hot start. Defaults to True.
        beta (double): Wilson action inverse coupling constant
        angle_l (double): total distance upon which we select a sample
        param_lambda_l (double): poisson law parameter of the total distance upon which we change the direction of updating
        param_pos_l (double): poisson law parameter of the total distance upon which we randomly change the updated link
        n (int, optional): number of samples to generate. Defaults to 5.

    Returns:
        list of double: list of actions of n samples of gauge configurations
    """
    sample_walker = ECMC_samples(L, T, cold, beta, angle_l, param_lambda_l, param_pos_l,n)
    actions = [calculate_action(i, beta) for i in sample_walker]
    return actions
