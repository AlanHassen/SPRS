""" Module containing classes and routines for making stock input to the tree search.
"""
from __future__ import annotations

import argparse
import importlib
from typing import TYPE_CHECKING

try:
    import molbloom
except ImportError:
    HAS_MOLBLOOM = False
else:
    HAS_MOLBLOOM = True

import pandas as pd
from rdkit import Chem

from aizynthfinder.chem import Molecule, MoleculeException
from aizynthfinder.context.stock import MongoDbInchiKeyQuery

if TYPE_CHECKING:
    from aizynthfinder.utils.type_utils import Iterable, List, Optional

    _StrIterator = Iterable[str]


def _get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("smiles2stock")
    parser.add_argument(
        "--files",
        required=True,
        nargs="+",
        help="the files containing smiles",
    )
    parser.add_argument(
        "--source",
        choices=["plain", "module"],
        help="indicates how to read the files. "
        "If 'plain' is used the input files should only contain SMILES (one on each row), "
        "if 'module' is used the SMILES are loaded from by python module"
        " (see documentation for details)",
        default="plain",
    )
    parser.add_argument(
        "--output",
        required=True,
        default="",
        help="the name of the output file or source tag",
    )
    parser.add_argument(
        "--target",
        choices=["hdf5", "mongo", "molbloom", "molbloom-inchi","csv"],
        help="type of output",
        default="hdf5",
    )
    parser.add_argument("--host", help="the host of the Mongo database")
    parser.add_argument(
        "--bloom_params", nargs=2, type=int, help="the parameters to the Bloom filter"
    )
    return parser.parse_args()


def _convert_smiles(smiles_list: _StrIterator) -> _StrIterator:
    for smiles in smiles_list:
        try:
            inchi = Molecule(smiles=smiles, sanitize=True).inchi_key
            yield (smiles, inchi)
        except MoleculeException:
            print(
                f"Failed to convert {smiles} to inchi key. Probably due to sanitation.",
                flush=True,
            )


def extract_plain_smiles(files: List[str]) -> _StrIterator:
    """
    Extract SMILES from plain text files, one SMILES on each line.
    The SMILES are yielded to save memory.
    """
    for filename in files:
        print(f"Processing {filename}", flush=True)
        with open(filename, "r") as fileobj:
            for line in fileobj:
                yield line.strip()


def extract_smiles_from_module(files: List[str]) -> _StrIterator:
    """
    Extract SMILES by loading a custom module, containing
    the function ``extract_smiles``.

    The first element of the input argument is taken as the module name.
    The other elements are taken as input to the ``extract_smiles`` method

    The SMILES are yielded to save memory.
    """
    module_name = files.pop(0)
    module = importlib.import_module(module_name)
    if not files:
        for smiles in module.extract_smiles():  # type: ignore
            yield smiles
    else:
        for filename in files:
            print(f"Processing {filename}", flush=True)
            for smiles in module.extract_smiles(filename):  # type: ignore
                yield smiles


def make_hdf5_stock(smiles_inchi_keys_gen: _StrIterator, filename: str) -> None:
    """
    Put all the inchi keys from the given iterable in a pandas
    dataframe and save it as an HDF5 file. Only unique inchi keys
    are stored.
    """
    data = pd.DataFrame(smiles_inchi_keys_gen, columns=["smiles", "inchi_key"])
    #data = pd.DataFrame.from_dict({"inchi_key": inchi_keys})
    #print(data.columns)
    data = data.drop_duplicates("inchi_key")
    data.to_hdf(filename, "table")
    print(f"Created HDF5 stock with {len(data)} unique compounds")

def make_csv_stock(smiles_inchi_keys_gen: _StrIterator, filename: str) -> None:
    """
    Put all the inchi keys from the given iterable in a pandas
    dataframe and save it as an HDF5 file. Only unique inchi keys
    are stored.
    """
    data = pd.DataFrame(smiles_inchi_keys_gen, columns=["smiles", "inchi_key"])
    print(data.columns)
    data = data.drop_duplicates("inchi_key")
    data.to_csv(filename, index=False)
    print(f"Created CSV stock with {len(data)} unique compounds")


def make_molbloom(
    smiles_list: _StrIterator, filename: str, filter_size: int, approx_mols: int
) -> None:
    """
    Put all the unique SMILES in a new bloom filter.

    :params smiles_list: the SMILES
    :params filename: the path to the saved filter
    :params filter_size: the size of the filter in bits
    :params approx_mols: approximately the number of compounds
    """
    filter_ = molbloom.CustomFilter(filter_size, approx_mols, "myfilter")
    processed_smiles = set()
    for smiles in smiles_list:
        try:
            smiles_can = Chem.CanonSmiles(smiles)
        # pylint: disable=broad-except
        except Exception:
            print(
                f"Failed to convert {smiles} to canonical SMILES.",
                flush=True,
            )
            continue
        if smiles_can in processed_smiles:
            continue
        filter_.add(smiles_can)
        processed_smiles.add(smiles_can)
    filter_.save(filename)
    print(f"Created bloom stock with {len(processed_smiles)} unique compounds")


def make_molbloom_inchi(
    inchi_keys: _StrIterator, filename: str, filter_size: int, approx_mols: int
) -> None:
    """
    Put all the unique InChI keys in a new bloom filter.

    :params inchi_keys: the Inchi Keys
    :params filename: the path to the saved filter
    :params filter_size: the size of the filter in bits
    :params approx_mols: approximately the number of compounds
    """
    filter_ = molbloom.CustomFilter(filter_size, approx_mols, "myfilter")
    nadded = 0
    for inchi_key in inchi_keys:
        filter_.add(inchi_key)
        nadded += 1
    filter_.save(filename)
    print(f"Created bloom stock with {nadded} unique compounds")


def make_mongo_stock(
    inchi_keys: _StrIterator, source_tag: str, host: Optional[str] = None
) -> None:
    """
    Put all the inchi keys from the given iterable in Mongo database as
    a molecules collection. Only unique inchi keys are stored.
    """
    mol_collection = MongoDbInchiKeyQuery(host=host).molecules
    if "inchi_key" not in mol_collection.index_information():
        mol_collection.create_index("inchi_key", name="inchi_key")
    mol_collection.delete_many({"source": source_tag})
    inchi_keys = set(inchi_keys)
    docs = ({"inchi_key": inchi_key, "source": source_tag} for inchi_key in inchi_keys)
    mol_collection.insert_many(docs)
    print(f"Created MongoDB stock with {len(inchi_keys)} unique compounds")


def main() -> None:
    """Entry-point for the smiles2stock tool"""
    args = _get_arguments()
    if args.source == "plain":
        smiles_gen = (smiles for smiles in extract_plain_smiles(args.files))
    else:
        smiles_gen = (smiles for smiles in extract_smiles_from_module(args.files))

    if not HAS_MOLBLOOM and args.target.startswith("molbloom"):
        raise ImportError(
            "Cannot create this stock format because it seems like molbloom is not installed. "
            "Please install aizynthfinder with extras dependencies."
        )

    if args.target == "molbloom":
        make_molbloom(smiles_gen, args.output, *args.bloom_params)
        return

    #inchi_keys_gen = (inchi_key for inchi_key in _convert_smiles(smiles_gen))
    smiles_inchi_keys_gen = ((smiles, inchi_key) for smiles, inchi_key in _convert_smiles(smiles_gen))
    inchi_keys_gen = "NOT IMPLEMENTED"
    if args.target == "hdf5":
        make_hdf5_stock(smiles_inchi_keys_gen, args.output)
    elif args.target == "csv":
        make_csv_stock(smiles_inchi_keys_gen, args.output)
    elif args.target == "molbloom-inchi":
        make_molbloom_inchi(inchi_keys_gen, args.output, *args.bloom_params)
    else:
        make_mongo_stock(inchi_keys_gen, args.output, args.host)


if __name__ == "__main__":
    main()
