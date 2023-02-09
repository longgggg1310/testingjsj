import numpy as np
from flask import Flask, request, render_template, abort
import pickle
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import Chem
import pubchempy
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from PIL import Image
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import wikipediaapi
import time


flask_app = Flask(__name__)
model = pickle.load(open("rf.pkl", "rb"))

@flask_app.errorhandler(404)
def function_name(error):
    return render_template('404.html'),404

@flask_app.route("/modal")
def modal():
    return render_template('modal.html', methods = ["POST"] )

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    smile = ''.join([str(x) for x in request.form.values()])
    try:
        formula = getSmile(smile)
        name_text = getName(formula)
        print('wiki', name_text)
        mol = AllChem.MolFromSmiles(smile)
        AllChem.Compute2DCoords(mol)
        Draw.MolToFile(mol,"static/smiles.png",size=(500,300))
        full_name = change_name(smile)
        float_features = smiles_to_fp(smile)
        features = [np.array(float_features)]
        prediction = model.predict(features)
        prediction= prediction.round(0).astype(int)
        final_prediction = 'ACTIVE' if prediction == 1 else 'INACTIVE'
        return render_template("result.html", prediction_text=final_prediction, 
                               name=smile, full_name=full_name, molecular=formula, 
                               name_text=name_text)
    except: 
        return render_template('404.html'), 404


def getSmile(smile):
    mol = Chem.MolFromSmiles(smile)
    formula = CalcMolFormula(mol)
    return formula
    
def getName(smile):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page_py = wiki_wiki.page(smile)

    ("Page - Exists: %s" % page_py.exists())
    a = ("%s" % page_py.title)
    return a

def smiles_to_fp(smiles, method="maccs", n_bits=2048):
    """
    Encode a molecule from a SMILES string into a fingerprint.

    Parameters
    ----------
    smiles : str
        The SMILES string defining the molecule.

    method : str
        The type of fingerprint to use. Default is MACCS keys.

    n_bits : int
        The length of the fingerprint.

    Returns
    -------
    array
        The fingerprint array.
    """

    # Convert smiles to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)

    if method == "maccs":
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    if method == "morgan2":
        return np.array(GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
    if method == "morgan3":
        return np.array(GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits))
    else:
        print(f"Warning: Wrong method specified: {method}." " Default will be used instead.")
        return np.array(MACCSkeys.GenMACCSKeys(mol))

def change_name(name):
    compounds = pubchempy.get_compounds(name, namespace='smiles')
    match = compounds[0]
    return match.iupac_name

if __name__ == "__main__":
    flask_app.run(debug=True)