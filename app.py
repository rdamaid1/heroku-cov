from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server

import pubchempy as pcp
from propy import PyPro
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.PyPro import GetProDes
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import pickle
import numpy as np
model = keras.models.load_model('model_sae_dnn')
app = Flask(__name__)


def predict():
    put_row([put_image('https://hukum.ipb.ac.id/wp-content/uploads/2021/05/Logo-IPB-University-Horizontal-1536x402.png'), None, put_markdown('### PUSAT STUDI BIOFARMAKA TROPIKA<br/>LPPM IPB')])
    
    ligand0 = input("Enter the PubChem ID of ligand: ", type=NUMBER)
    target0 = input("Enter the UniProt ID of protein target: ", type=TEXT)

    ligand = pcp.Compound.from_cid(ligand0)
    temp = bin(int(ligand.fingerprint, 16))
    fp = temp[2:883]
    list0 = list(fp)
    for i in range(len(list0)):
        list0[i] = float(list0[i])
    
    # download the protein sequence by uniprot id
    proteinsequence = GetProteinSequence(target0)

    target = GetProDes(proteinsequence).GetDPComp()
    target = [*target.values()]

    ligand_protein = list0 + target

    input0 = pd.DataFrame(ligand_protein).transpose()

    prediction = model.predict([[input0]])
    output = np.round(prediction[0][0], 10)

    if output < 0:
        put_text("Sorry, the ligand protein interaction cannot be predicted")

    else:
        put_text('Your ligand protein interaction prediction score is:', output*100, "%")
        
    # Year = input("Enter the Model Year：", type=NUMBER)
    # Year = 2021 - Year
    # Present_Price = input("Enter the Present Price(in LAKHS)", type=FLOAT)
    # Kms_Driven = input("Enter the distance it has travelled(in KMS)：", type=FLOAT)
    # Kms_Driven2 = np.log(Kms_Driven)
    # Owner = input("Enter the number of owners who have previously owned it(0 or 1 or 2 or 3)", type=NUMBER)
    # Fuel_Type = select('What is the Fuel Type', ['Petrol', 'Diesel','CNG'])
    # if (Fuel_Type == 'Petrol'):
    #     Fuel_Type = 239

    # elif (Fuel_Type == 'Diesel'):
    #     Fuel_Type = 60

    # else:
    #     Fuel_Type = 2
    # Seller_Type = select('Are you a dealer or an individual', ['Dealer', 'Individual'])
    # if (Seller_Type == 'Individual'):
    #     Seller_Type = 106

    # else:
    #     Seller_Type = 195
    # Transmission = select('Transmission Type', ['Manual Car', 'Automatic Car'])
    # if (Transmission == 'Manual Car'):
    #     Transmission = 261
    # else:
    #     Transmission = 40


app.add_url_rule('/tool', 'webio_view', webio_view(predict),
            methods=['GET', 'POST', 'OPTIONS'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(predict, port=args.port)
#if __name__ == '__main__':
    #predict()

#app.run(host='localhost', port=80)

#visit http://localhost/tool to open the PyWebIO application.
