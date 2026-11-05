import numpy as np 
from matplotlib import pyplot as plt
import astropy.constants as ac
from utils import reflect_across_line, shift_reference_system, transform_image_to_matrix, rescale_matrix, rotation_matrix
import pandas as pd
from tqdm import tqdm
import scipy.optimize
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d
from matplotlib import patches

def parametric_straylights_x(theta,iota,gamma,xi,delta,epsilon,r0,beta0,FocalLenght,phi):\
    
    '''
        The function calculates the x coordinate of the stray light photons on the plane following Spiga (2015) for a 
        Walter telescope with Focal length = FocalLenght. This function includes only the possible inclination
        of the hyperbola since the only the hyperbola straylight is considered.
    '''

    thetap = np.sqrt(theta**2 + (delta/FocalLenght)**2 - 2*theta*(delta/FocalLenght)*np.cos(epsilon - iota))
    iotap = np.arctan2((theta*np.sin(iota) + (delta/FocalLenght)*np.sin(epsilon)),(theta*np.cos(iota) + (delta/FocalLenght)*np.cos(epsilon)))

    x = (r0 - 2*beta0*FocalLenght)*np.cos(phi) + thetap*FocalLenght*np.cos(2*phi) + (np.rad2deg(gamma))*FocalLenght*(np.cos(2*phi - (xi - iotap)) + np.cos(xi-iotap))
    #x = (r0 - 2*beta0*FocalLenght)*np.cos(phi-iota) + theta*FocalLenght*np.cos(2*phi-2*iota) + gamma*FocalLenght*(np.cos(2*phi + xi - iota) + np.cos(xi-iota))
    
    return x


def parametric_straylights_y(theta,iota,gamma,xi,delta,epsilon,r0,beta0,FocalLenght,phi):\
    
    '''
        The function calculates the x coordinate of the stray light photons on the plane following Spiga (2015) for a 
        Walter telescope with Focal length = FocalLenght. This function includes only the possible inclination
        of the hyperbola since the only the hyperbola straylight is considered.
    '''


    thetap = np.sqrt(theta**2 + (delta/FocalLenght)**2 - 2*theta*(delta/FocalLenght)*np.cos(epsilon - iota))
    iotap = np.arctan2((theta*np.sin(iota) + (delta/FocalLenght)*np.sin(epsilon)),(theta*np.cos(iota) + (delta/FocalLenght)*np.cos(epsilon)))

    y = (r0 - 2*beta0*FocalLenght)*np.sin(phi) + thetap*FocalLenght*np.sin(2*phi) + (np.rad2deg(gamma))*FocalLenght*(np.sin(2*phi - (xi - iotap)) + np.sin(xi-iotap))
    #y = (r0 - 2*beta0*FocalLenght)*np.sin(phi-iota) + theta*FocalLenght*np.sin(2*phi-2*iota) + gamma*FocalLenght*(np.sin(2*phi + xi - iota) + np.sin(xi-iota))
    
    return y

    
def analytic_arcs(nshell,misparams,polar_coords, \
                         phi,r0,FocalLength,chord_correction_factor=0.1,focal_correction_factor=0.1):
        
        
    '''
        This function reproduces the straylight pattern through analytical approach (Spiga 2015) for 
        XMM-Newton mirrors. The pattern depends by two main parameters which are the source distance theta and 
        the source position in the polar plane iota. The function manages to take into account possible misalignments of 
        the hyperbolas described by amplitude (dtheta0) and direction (xi). The implementation for the 
        parabola is straight forward.
           
        nshell: follows the array shell number --> nshell = 0 corresponds to the first shell. Order of the shells are 
                given from the outermost to the innermost following the XMM-Newton convention but it depends on the input radius

        misparams: array of misalignment parameters [gamma,xi,delta,epsilon] where gamma is the amplitude of the misalignment, xi is the direction of the misalignment, 
                delta is the distance between the optical axis and the physical center of the detector and epsilon 
                is the position angle of the optical axis.

        polar_coords: array of polar coordinates [theta,iota] where theta is the distance between the source and the optical axis and iota is the position angle of the source.
                      This coordinates are meant to be obtained by the ecoordconv task from sas, meaning these are the coordinates
                      of he TelCoord reference system.

        phi: array of azimuthal angles of the mirror shell. The function will return the straylight pattern for each azimuthal angle in phi.
             They are the angles which point of the aximuthal position of the photons

        r0: radius of the mirror shell --> must be an array of the same length of the number of shells

        chord_correction_factor: conversion factor from the double cone approximation

        focal_correction_factor: conversion factor from the double cone approximation

        Angles unit of measure: all angles are expressed in radians

        The last part of the function shifts the pattern according to the reference system one chose. The reference frame
        of the optics has the origin in the intersection between the optical axis and the detector. The variable quadrant 
        defines where the optical axis is according to the position angle value. 

        Example: a position angle of around 270º corresponds to the optical axis being in the fourth quadrant

        x_coord_opt,y_coord_opt: distance between the position of the optical axis and the physical center of the detector
    '''

    # defining the misalignment parameters
    gamma,xi,delta,epsilon = misparams

    # defining the polar coordinates of the source
    theta,iota = polar_coords
    

    #calcolo le quantità relative al doppio cono
    r = r0[nshell]
    alfa = np.arctan(r/FocalLength) / 4
    beta0 = 3*alfa

    #defining correction according to double cone approximation
    dbeta1 = chord_correction_factor*beta0
    dbeta2 = focal_correction_factor*beta0

    #new values
    beta = beta0 + dbeta1
    f = FocalLength*np.tan(beta0)/np.tan(beta + dbeta2)

    x = parametric_straylights_x(theta,iota,gamma,xi,delta,epsilon,r,beta,f,phi)
    y = parametric_straylights_y(theta,iota,gamma,xi,delta,epsilon,r,beta,f,phi)


    iotap = np.arctan2((theta*np.sin(iota) + (delta/FocalLength)*np.sin(epsilon)),(theta*np.cos(iota) + (delta/FocalLength)*np.cos(epsilon)))

    #Sposto il pattern nella posizione opposta a iota
    iota_rel = iotap - np.pi
    x_sim, y_sim = rotation_matrix(x,y,iota_rel)

    #faccio l'immagine specchiata del pattern
    x_fin, y_fin = reflect_across_line(x_sim, y_sim, iota)
    #x_sim = x
    #y_sim = y
    
    return x_fin,y_fin
