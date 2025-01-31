import functools
import inspect
import os
import pickle
import zlib
from collections import defaultdict
from copy import copy
from datetime import datetime
from functools import cache
from pathlib import Path
from pickle import UnpicklingError
from time import time

import numpy as np
import pandas as pd
from colorama import Fore


# from colorama import Fore


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    if isinstance(d, dict):
        for key, d_sub in d.items():
            d[key] = defaultdict_to_dict(d_sub)
    if isinstance(d, list):
        d = np.array(d)
    return d


def getVariableName(variable, globalVariables):
    # from: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
    """ Get Variable Name as String by comparing its ID to globals() Variables' IDs
        args:
            variable(var): Variable to find name for (Obviously this variable has to exist)
        kwargs:
            globalVariables(dict): Copy of the globals() dict (Adding to Kwargs allows this function to work properly when imported from another .py)
    """
    for globalVariable in globalVariables:
        if id(variable) == id(
                globalVariables[globalVariable]):  # If our Variable's ID matches this Global Variable's ID...
            return globalVariable  # Return its name from the Globals() dict


def obj2str(val):
    kwargs_str = ''
    if isinstance(val, np.ndarray):
        return ''
    elif isinstance(val, dict):
        for key2 in sorted(val.keys()):
            kwargs_str += obj2str(val[key2])

    elif isinstance(val, list):
        for val_ in val:
            kwargs_str += obj2str(val_)

    else:
        kwargs_str += f'{val}_'
    if len(kwargs_str) > 20:
        kwargs_str = str(zlib.adler32(kwargs_str.encode()))

    return kwargs_str


def f2str(callback, kwargs=None):
    if kwargs is None:
        kwargs = {}
    signature = inspect.signature(callback)  # functools.partial impacts sig
    for k, v in signature.parameters.items():
        if v.default is v.empty: continue  # exclude args, only want kwargs
        if k not in kwargs: kwargs[k] = v.default  # TODO: could toggle off?
    if kwargs is not None:
        kwargs_str = ''
        for key in sorted(kwargs.keys()):
            val = kwargs[key]
            kwargs_str += obj2str(val)
        kwargs_str = kwargs_str[:-1]
    return kwargs_str


def get_default_fp(args, kwargs, callback, cache_dir, verbose=0):
    if verbose > 0: print(f'Making default filepath: {kwargs=}')
    if args is not None:
        args_str = '_'.join(args)
    else:
        args_str = ''
    kwargs_str = f2str(callback, kwargs)
    if isinstance(callback, functools.partial):
        name = callback.func.__name__
    else:
        name = callback.__name__
    func_dir = f'{cache_dir}/{name}'
    Path(func_dir).mkdir(parents=True, exist_ok=True)
    filepath = f'{func_dir}/{args_str}_{kwargs_str}.pkl'
    if verbose > 0: print(f'Default pickle_wrap filepath: {filepath}')
    return filepath


PICKLE_CACHE = {}


def pickle_wrap(callback: object, filepath: object = None,
                args: object = None, kwargs: object = None,
                easy_override: object = False,
                verbose: object = 0, cache_dir: object = 'cache',
                dt_max: object = None, RAM_cache: object = False):
    '''
    :param filepath: File to which the callback output should be loaded
                     (if already created)
                     or where the callback output should be saved
    :param callback: Function to be pickle_wrapped
    :param args: Arguments passed to function (often not necessary)
    :param kwargs: Kwargs passed to the function (often not necessary)
    :param easy_override: If true, then the callback will be performed and the
                          previous .pkl save will be overwritten (if it exists)
    :param verbose: If true, then some additional details will be printed
                    (the name of the callback, and the time needed to perform
                    the function or load the .pkl)
    :return: Returns the output of the callback or the output saved in filepath
    '''
    kwargs = copy(kwargs)  # don't want to modify outside
    if filepath is None:
        filepath = get_default_fp(args, kwargs, callback, cache_dir, verbose)
    if RAM_cache and filepath in PICKLE_CACHE:
        return PICKLE_CACHE[filepath]

    if verbose > 0:
        print(f'pickle_wrap: {filepath=}')
        print('\tFunction:', getVariableName(callback,
                                             globalVariables=globals().copy()))

    if os.path.isfile(filepath):
        made = os.path.getmtime(filepath)
        fn = os.path.basename(filepath)
        if dt_max is None:
            dt_max = datetime(2023, 11, 18, 14, 0, 0, 0)
        dt_min = datetime(2023, 11, 1, 14, 0, 0, 0)

        dt = datetime.fromtimestamp(made)
        if dt < dt_max and dt > dt_min:
            easy_override = True
            if verbose >= 0:
                print(f'File ({fn}) was made: {made}')
                print('\tFile is old, overriding')

    if os.path.isfile(filepath) and not easy_override:
        try:
            start = time()
            with open(filepath, "rb") as file:
                pk = pickle.load(file)
                if verbose > 0: print(f'\tLoad time: {time() - start:.5f} s')
                if verbose == 0: print(f'Pickle loaded '
                                       f'({time() - start:.3f} s): '
                                       f'{filepath=}')
                if RAM_cache: PICKLE_CACHE[filepath] = pk
                return pk
        except (UnpicklingError, MemoryError, EOFError) as e:
            print(f'{Fore.RED}{e=}')
            print(f'\t{Fore.YELLOW}{callback=}')
            print(f'\t{Fore.YELLOW}{filepath=}{Fore.RESET}')

    if verbose > 0:
        print('Callback:',
              getVariableName(callback, globalVariables=globals().copy()))
    start = time()
    if args:
        output = callback(*args)
    elif kwargs:
        output = callback(**kwargs)
    else:
        output = callback()
    if verbose > 0:
        print(f'\tFunction time: {time() - start:.3f} s')
        # print('\tDumping to file name:', filepath)
    start = time()
    try:
        # print(f'{filepath=}')

        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
        if verbose == 0: print(f'Pickle wrapped ({time() - start:.3f} s): '
                               f'{filepath=}')
    except FileNotFoundError:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as new_file:
            pickle.dump(output, new_file)
    if verbose > 0: print(f'\tDump time: {time() - start:.3f} s')
    if RAM_cache: PICKLE_CACHE[filepath] = output

    return output


def get_formula_cols(df, formula):
    import re
    formula = re.split(' |[*]', formula)
    cols = []
    for col in df.columns:
        if col in formula:
            cols.append(col)
    return cols


def save_csv_pkl(df, fp_csv, no_csv=False, check_dup=True, verbose=1):
    if check_dup and df['doi_str'].duplicated().any():
        raise ValueError(f'There are duplicated DOIs. '
                         f'{df["doi_str"].value_counts()}')
    nrows = len(df)
    ncol = len(df.columns)
    if not no_csv:
        if verbose: print(f'Saving df with {nrows:,} rows and {ncol} columns')
        t_st = time()
        df.to_csv(fp_csv, index=False)
        size = get_size_str(os.path.getsize(fp_csv) / 1e6)
        t_end = time()
        if verbose: print(f'\tTime needed for .csv ({size}): '
                          f'{t_end - t_st:.3f} s')
    fn_csv = os.path.basename(fp_csv)
    fp_pkl = f'../cache/{fn_csv}.pkl'
    t_st = time()
    if verbose: print(f'\tOnto saving the pkl: {fn_csv}')
    with open(fp_pkl, 'wb') as file:
        pickle.dump(df, file)
    t_end = time()
    size = get_size_str(os.path.getsize(fp_pkl) / 1e6)
    if verbose: print(f'\tTime needed for .pkl ({size}): {t_end - t_st:.3f} s')


def get_size_str(MB):
    if MB > 1e3:
        size = f'{MB / 1e3:.1f} GB'
    else:
        size = f'{MB:.1f} MB'
    return size


@cache
def read_csv_fast(fp_csv, easy_override=False, check_dup=True, verbose=1):
    if 'by_p' in fp_csv: check_dup = False
    fn_csv = os.path.basename(fp_csv)
    fp_pkl = f'../cache/{fn_csv}.pkl'
    if os.path.isfile(fp_pkl) and not easy_override:
        t_st = time()
        with open(fp_pkl, 'rb') as file:
            df = pickle.load(file)
        t_end = time()
        nrows = len(df)
        ncol = len(df.columns)
        size = get_size_str(os.path.getsize(fp_pkl) / 1e6)
        if verbose:
            print(f'Read csv .pkl with {nrows:,} rows and {ncol} columns: '
                  f'{fn_csv}')
            print(f'\tTime needed to read .pkl ({size}): {t_end - t_st:.3f} s')
        if check_dup and df['doi_str'].duplicated().any():
            print(f'{fp_csv=}')
            cnt = df["doi_str"].value_counts(dropna=False).sort_values()
            raise ValueError(f'There are duplicated DOIs. {cnt}')
        return df

    else:
        t_st = time()
        df = pd.read_csv(fp_csv)
        size = get_size_str(os.path.getsize(fp_csv) / 1e6)
        t_end = time()
        nrows = len(df)
        ncol = len(df.columns)
        if verbose:
            print(f'Read .csv with {nrows:,} and {ncol} columns: {fn_csv}')
            print(f'\tTime needed to read .csv ({size}): {t_end - t_st:.3f} s')

        t_st = time()
        with open(fp_pkl, 'wb') as file:
            pickle.dump(df, file)
        t_end = time()
        size = get_size_str(os.path.getsize(fp_pkl) / 1e6)
        if verbose:
            print(f'\tSaved .pkl version ({size}) in: {t_end - t_st:.3f} s')
        if (check_dup and 'doi_str' in df.columns and
                df['doi_str'].duplicated().any()):
            cnt = df["doi_str"].value_counts(dropna=False).sort_values()
            raise ValueError(f'There are duplicated DOIs. {cnt}')
        return df


PSYC_SUBJECTS_ = ['Psychology_Miscellaneous',
                  'Applied_Psychology',
                  'General_Psychology',
                  'Developmental_and_Educational_Psychology',
                  'Clinical_Psychology',
                  'Social_Psychology',
                  'Experimental_and_Cognitive_Psychology',
                  ]
PSYC_SUBJECTS_space = [  # 'Psychology miscellaneous', not right
    'Psychology (Miscellaneous)',
    'Applied Psychology',
    'General Psychology',
    'Developmental and Educational Psychology',
    'Clinical Psychology',
    'Social Psychology',
    'Experimental and Cognitive Psychology',
]
