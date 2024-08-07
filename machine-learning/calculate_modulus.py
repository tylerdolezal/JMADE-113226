import glob
import numpy as np
from ase.io.vasp import read_vasp

def debye_temp(b, g):

    first = glob.glob("finished/structures/*")
    first.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    second = glob.glob("training_scripts/training_structures/*")
    second.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    first = np.concatenate((first,second))

    #first = glob.glob("mc2-results/structures/*")
    #first.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    debye = []

    for afile, B, G in zip(first, b, g):
        pos = read_vasp(afile)
        # The Planck's constant in m^2 Kg s^-1
        h = 6.626E-34
        # The reduced Planck's constant hbar in m^2 Kg s^-1
        hbar = 1.0546E-34
        # The Boltzmann constant in m^2 Kg s^-2 K-1
        k = 1.381E-23
        # The Avogadro's Constant
        Na = 6.02E+23
        # The total volume of the supercell (molecule)
        volume = pos.get_volume()
        # The total mass of all atoms in the supercell (molecule), in AMU
        M = sum(pos.get_masses())
        # The number of atoms in the supercell (molecule)
        n = pos.get_global_number_of_atoms()

        # The density in Kg/m^3
        rho = M*1E-3/Na/volume/1E-30

        V_s = 1E-3*G**0.5*(1E+9/(M*1E-3/volume/1E-30/(6.02E+23)))**0.5
        V_b = 1E-3*B**0.5*(1E+9/(M*1E-3/volume/1E-30/(6.02E+23)))**0.5
        V_p = 1E-3*(B+(4.*G/3.))**0.5*(1E+9/(M*1E-3/volume/1E-30/(6.02E+23)))**0.5
        V_m = (((2./V_s**3.) + (1./V_p**3))/3.)**(-1./3.)
        T_D = h/k*((3.*n/(4.*np.pi))*Na*rho/M/1E-3)**(1./3.)*V_m*1E+3
        debye.append(T_D)

    return(np.array(debye))


def modulus(y_matrix):
    """
    c11 = y_matrix[:,0]
    c12 = y_matrix[:,1]
    c13 = y_matrix[:,2]
    c14 = y_matrix[:,3]
    c15 = y_matrix[:,4]
    c16 = y_matrix[:,5]
    c22 = y_matrix[:,6]
    c23 = y_matrix[:,7]
    c24 = y_matrix[:,8]
    c25 = y_matrix[:,9]
    c26 = y_matrix[:,10]
    c33 = y_matrix[:,11]
    c34 = y_matrix[:,12]
    c35 = y_matrix[:,13]
    c36 = y_matrix[:,14]
    c44 = y_matrix[:,15]
    c45 = y_matrix[:,16]
    c46 = y_matrix[:,17]
    c55 = y_matrix[:,18]
    c56 = y_matrix[:,19]
    c66 = y_matrix[:,20]
    """
    c11 = y_matrix[:,0]
    c12 = y_matrix[:,1]
    c13 = y_matrix[:,2]
    c14 = 0.0
    c15 = 0.0
    c16 = 0.0
    c22 = y_matrix[:,3]
    c23 = y_matrix[:,4]
    c24 = 0.0
    c25 = 0.0
    c26 = 0.0
    c33 = y_matrix[:,5]
    c34 = 0.0
    c35 = 0.0
    c36 = 0.0
    c44 = y_matrix[:,6]
    c45 = 0.0
    c46 = 0.0
    c55 = y_matrix[:,7]
    c56 = 0.0
    c66 = y_matrix[:,8]
    

    a = c33*c55-c35*c35
    b = c23*c55-c25*c35
    c = c13*c35-c15*c33
    d = c13*c55-c15*c35
    e = c13*c25-c15*c23
    f = c11*(c22*c55-c25*c25)-c12*(c12*c55-c15*c25)+c15*(c12*c25-c15*c22)+c25*(c23*c35-c25*c33)
    g = c11*c22*c33-c11*c23*c23-c22*c13*c13-c33*c12*c12+2*c12*c13*c23
    O = 2*(c15*c25*(c33*c12-c13*c23)+c15*c35*(c22*c13-c12*c23)+c25*c35*(c11*c23-c12*c13))-(c15*c15*(c22*c33-c23*c23)+c25*c25*(c11*c33-c13*c13)+c35*c35*(c11*c22-c12*c12))+g*c55

    B_v = (c11+c22+c33+ 2*(c12+c13+c23))/9.
    G_v = (c11+c22+c33+3*(c44+c55+c66)-(c12+c13+c23))/15.
    B_r = O/(a*(c11+c22-2*c12)+b*(2*c12-2*c11-c23)+c*(c15-2*c25)+d*(2*c12+2*c23-c13-2*c22)+2*e*(c25-c15)+f)
    G_r = 15/(4*(a*(c11+c22+c12)+b*(c11-c12-c23)+c*(c15+c25)+d*(c22-c12-c23-c13)+e*(c15-c25)+f)/O+3*(g/O+(c44+c66)/(c44*c66-c46*c46)))

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    cp = c12 - c44
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    T_D = debye_temp(B_vrh, G_vrh)
    
    kleinman = (c11 + 8*c12) / (7*c11 + 2*c12)

    values = np.vstack((B_vrh, G_vrh, E, cp, v, T_D, kleinman))
    return(values.T)
