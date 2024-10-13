from django.shortcuts import render, redirect
from pycalphad import Database, equilibrium, variables as v
import numpy as np

# Load the Fe-Mn-C thermodynamic database
dbf = Database('FeMnC-11Dju.tdb')

# Thermodynamic function to calculate retained austenite fraction
def calculate_retained_austenite(annealing_temp, mn_mole_fraction, c_mole_fraction):
    comps = ['FE', 'C', 'MN', 'VA']
    phases = ['FCC_A1', 'BCC_A2']

    # Perform equilibrium calculation using pycalphad
    conditions = {v.X('MN'): mn_mole_fraction, v.X('C'): c_mole_fraction, v.T: annealing_temp, v.P: 101325}  # Pressure in Pa (1 atm)
    eq_result = equilibrium(dbf, comps, phases, conditions, output='GM')

    # Check for the presence of FCC_A1 phase (retained austenite)
    fcc_phase = eq_result.where(eq_result.Phase == 'FCC_A1', drop=True)

    if not fcc_phase.NP.isnull().all():
        mn_mole_fraction_fcc = fcc_phase.X.sel(component='MN').values.item()
        c_mole_fraction_fcc = fcc_phase.X.sel(component='C').values.item()
        f_gamma = fcc_phase.NP.sum().values.item()  # Sum of phase fractions of FCC_A1
        
        # Atomic weights (g/mol)
        atomic_weight_fe = 55.845
        atomic_weight_mn = 54.938
        atomic_weight_c = 12.011

        # Calculate mole fraction of Fe
        fe_mole_fraction_fcc = 1 - mn_mole_fraction_fcc - c_mole_fraction_fcc

        # Calculate total mass of the alloy in terms of atomic weights and mole fractions
        total_mass = (mn_mole_fraction_fcc * atomic_weight_mn) + (c_mole_fraction_fcc * atomic_weight_c) + (fe_mole_fraction_fcc * atomic_weight_fe)

        # Calculate weight percentages
        mn_weight_percent = (mn_mole_fraction_fcc * atomic_weight_mn / total_mass) * 100
        c_weight_percent = (c_mole_fraction_fcc * atomic_weight_c / total_mass) * 100

        # Koistinen-Marburger Equation for Ms and martensite fraction
        Ms = 812 - (423 * c_weight_percent) - (30.4 * mn_weight_percent)
        TQ = 298  # Quenching Temperature

        # Calculate martensite fraction
        f_prime = 1 - np.exp(-0.011 * (Ms - TQ))

        # Retained Austenite Fraction (RA)
        retained_austenite = f_gamma * (1 - f_prime)
        return retained_austenite * 100  # return as a percentage
    else:
        return None

import numpy as np

# Function to calculate the thermal diffusivity
def calculate_alpha(K, density, cp):
    return K / (density * cp)

# Function to calculate temperature distribution using Rosenthal equation
def temperature_distribution(X, Y, Z, P, K, v, alpha, T_0, t):
    R_t = np.sqrt((X / 1000 - v * t) ** 2 + Y * Y / 1000000 + Z * Z / 1000000)
    c1 = P / (2 * np.pi * K)
    c2 = v / (2 * alpha)
    T_t = T_0 + c1 * (np.exp(-c2 * (X / 1000 - v * t + R_t))) / R_t
    return T_t

# Function to evaluate printability based on melt pool dimensions
def evaluate_printability(L, W, D, t):
    if D / t < 1.5:
        return "Lack of Fusion"
    elif L / W > 2.3:
        return "Balling"
    elif W / D < 1.5:
        return "Keyhole Formation"
    else:
        return "Feasible Printability"

# User inputs
def melt_pool_dimensions_and_printability(P, K, v, density, cp, t, T_0, melting_point):
    # P = float(input("Enter the laser power (W): "))  # Laser power
    # K = float(input("Enter the thermal conductivity (W/mK): "))  # Thermal conductivity
    # v = float(input("Enter the scan speed (m/s): "))  # Laser scan speed
    # density = float(input("Enter the material density (kg/m³): "))  # Material density
    # cp = float(input("Enter the specific heat (J/kgK): "))  # Specific heat
    # t = float(input("Enter the powder layer thickness (m): "))  # Powder layer thickness
    # T_0 = float(input("Enter the initial temperature (K): "))  # Initial temperature
    # melting_point = float(input("Enter the melting point (K): "))  # Melting point

    # Constants and initial setup
    gridnumber = 300  # Grid resolution
    gridnumber_with_a_j = 300j
    X, Y, Z = np.mgrid[-10:10:gridnumber_with_a_j, -4:4:gridnumber_with_a_j, -1:0:gridnumber_with_a_j]
    alpha = calculate_alpha(K, density, cp)  # Thermal diffusivity

    # Calculate temperature distribution at t=0
    T_t = temperature_distribution(X, Y, Z, P, K, v, alpha, T_0, t=0)

    # Cap temperature at melting point for easier calculation
    MP = np.ones((gridnumber, gridnumber, gridnumber)) * melting_point
    T_t = np.fmin(T_t, MP)  # Cap temperatures at melting point
    T_t = np.nan_to_num(T_t)

    # Assume melt pool dimensions are based on temperature distribution
    # Here is an updated method to calculate L, W, D based on temperature profile
    L = np.max(X[T_t >= melting_point]) - np.min(X[T_t >= melting_point])  # Length
    W = np.max(Y[T_t >= melting_point]) - np.min(Y[T_t >= melting_point])  # Width
    D = np.max(Z[T_t >= melting_point]) - np.min(Z[T_t >= melting_point])  # Depth

    # Evaluate printability based on melt pool dimensions and thresholds
    printability = evaluate_printability(L, W, D, t)

    # Output the results
    # print(f"Melt Pool Dimensions (L={L:.3f} m, W={W:.3f} m, D={D:.3f} m)")
    # print(f"The process with laser power {P} W and scan speed {v} m/s results in: {printability}")
    return (L, W, D, printability)

def predictor(request):
    if request.method == 'POST':
        # Retrieve user inputs
        try:
            annealing_temp = float(request.POST['Annealing_Temperature'])
            c_mole_fraction = float(request.POST['Carbon_Content'])
            mn_mole_fraction = float(request.POST['Manganese_Content'])

            laser_power = float(request.POST['Laser_Power'])
            thermal_conductivity = float(request.POST['Thermal_Conductivity'])
            scan_speed = float(request.POST['Scan_Speed'])
            density = float(request.POST['Density'])
            specific_heat = float(request.POST['Specific_Heat'])
            powder_layer_thickness = float(request.POST['Powder_Layer_Thickness'])
            initial_temperature = float(request.POST['Initial_Temperature'])
            melting_point = float(request.POST['Melting_Point'])

            # Call the retained austenite calculation function
            retained_austenite_fraction = calculate_retained_austenite(annealing_temp, mn_mole_fraction, c_mole_fraction)

            # Call the melt pool dimensions and printability function
            L, W, D, printability = melt_pool_dimensions_and_printability(
                laser_power, thermal_conductivity, scan_speed, density,
                specific_heat, powder_layer_thickness, initial_temperature,
                melting_point
            )
            if retained_austenite_fraction is not None:
                result = {
                    "valid": True,
                    "austeniteFraction": f"{retained_austenite_fraction:.3f}",
                    "meltPoolDimensions": f"L: {L:.3f} m, W: {W:.3f} m, D: {D:.3f} m",
                    "printabilityCondition": printability

                }
            else:
                result = {
                    "valid": False,
                    "error": f"FCC_A1 phase not stable at {annealing_temp}K with Mn mole fraction {mn_mole_fraction} and C mole fraction {c_mole_fraction}.",
                    "meltPoolDimensions": f"L: {L:.3f} m, W: {W:.3f} m, D: {D:.3f} m",
                    "printabilityCondition": printability

                }
        except:
            result = {
                "valid": False,
                "error": "Please enter correct values.",
            }

        # Store the results in the session to pass to the next request
        request.session['result'] = result
        request.session['inputs'] = request.POST

        return redirect('predictor')  # Use the URL name for your view

    # On a GET request, retrieve the session data
    result = request.session.get('result', {})
    inputs = request.session.get('inputs', {})
    request.session['result'] = {}
    request.session['inputs'] = {}

    return render(request, 'main.html', {"result": result, "form_data": inputs})

