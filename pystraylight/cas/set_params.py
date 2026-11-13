import pandas as pd
import os
import argparse

if __name__ == "__main__":
    """
    Example of run of the module:

    python set_params.py --p_min 10 --p_max 50 --s_space 2 --m 5 --n_iter_dil 3 --xp 100 --yp 200 --filename DataBase_CAS.csv
    """
    parser = argparse.ArgumentParser(description="Addition of parameters to create a pandas database with configurations for CAS application")

    
    parser.add_argument("--filename", default="DataBase_CAS.csv", type=str,
                        help="The database must have specific columns to work out")

    # Mandatory Params
    parser.add_argument("--l_ratio", required=True, type=float, help="minimum threshold for gradients: p_min")
    parser.add_argument("--p_max", required=True, type=float, help="maximum threshold for gradients: p_max")
    parser.add_argument("--s_space", required=True, type=float, help="sigma for gaussian smoothing: s_space")
    parser.add_argument("--m", required=True, type=float, help="min_size parameter: m")
    parser.add_argument("--n_iter_dil", required=True, type=int, help="number of iteration for dilation procedure: n_iter_dil")
    parser.add_argument("--xp", required=True, type=float, help="X coordinate in pixel of the center of the image: xp")
    parser.add_argument("--yp", required=True, type=float, help="Y coordinate in pixel of the center of the image: yp")

    # Parametri facoltativi con valori di default (es. [5, 3, 2] + gamma)
    parser.add_argument("--connectivity", default=5, type=int, help="Parametro opzionale connectivity (default: 5)")
    parser.add_argument("--ksize", default=3, type=int, help="Size of the kernels ")
    parser.add_argument("--zoom", default=300.0, type=float, help="Size (in pixel) of the zoom for the image ")
    parser.add_argument("--gamma", default=2.0, type=float, help="Parametro opzionale gamma (default: 0)")

    args = parser.parse_args()

    filename = args.filename

    # Verifica se il database esiste già per calcolare il numero della configurazione
    if os.path.exists(filename):
        database = pd.read_csv(filename)
        conf_index = len(database) + 1
    else:
        # Se non esiste, creiamo un DataFrame vuoto con le colonne corrette
        database = pd.DataFrame(columns=[
            "Configuration", "l_ratio", "p_max", "s_space", "m", 
            "n_iter_dil", "connectivity", "ksize", "xp", "yp", "zoom", "gamma"
        ])
        conf_index = 1

    # Creazione della nuova riga con il nome della configurazione in prima colonna
    new_row = {
        "Configuration": f"Conf{conf_index}",
        "l_ratio": args.l_ratio,
        "p_max": args.p_max,
        "s_space": args.s_space,
        "m": args.m,
        "n_iter_dil": args.n_iter_dil,
        "connectivity": args.connectivity,
        "ksize": args.ksize,
        "xp": args.xp,
        "yp": args.yp,
        "zoom": args.zoom,
        "gamma": args.gamma
    }

    # Aggiunta della nuova riga al database esistente
    database = pd.concat([database, pd.DataFrame([new_row])], ignore_index=True)

    # Esportazione/Aggiornamento del file CSV
    database.to_csv(filename, index=False)

    print(f"\nConfigurazione 'Conf{conf_index}' aggiunta con successo a '{filename}'!")
    print("\nDatabase aggiornato:")
    print(database)