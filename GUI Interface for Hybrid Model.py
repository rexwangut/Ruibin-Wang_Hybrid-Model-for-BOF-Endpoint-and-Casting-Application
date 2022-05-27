import tkinter
from tkinter.filedialog import askopenfile
import tkinter.ttk
import numpy as np
import tkinter.font
from tensorflow.keras.models import load_model
import pandas as pd
from pickle import load
import csv
from csv import writer
from csv import reader
from tkinter import *
import os
from slagmodel import *
from TheoreticalT import *
from TheoreticalC import theo_c
from TheoreticalP import *
from HybridModel import *
from tensorflow import *

os.environ["SDL_VIDEO_CENTERED"] = "1"

#File Upload Window
def filewindow():

    newWindow1 = tkinter.Toplevel(window)

    newWindow1.title("File Upload")

    # sets the geometry of toplevel (File Upload Window size)
    newWindow1.geometry("800x200")

    def open_file():
        global x_input_temp
        global x_input_carbon
        global x_input_phos
        global x_input_temp_con
        global x_input_carbon_con
        global x_input_phos_con
        global y
        global predT
        global result
        global file_path
        global data
        file_path = askopenfile(mode='r', filetypes=[('CSV', '.csv')]).name
        if file_path is None:
            return
        df = open(file_path)
        data = pd.read_csv(df)
        data.dropna(subset=['HMA_S', 'HMA_CR'], inplace=True)
        x_input_temp = data[
            ['HMA_C', 'HMA_MN', 'HMA_P', 'HMA_S', 'HMA_SI', 'HMA_CR', 'HMA_TI',
                        'BLOW_DUR', 'HMWT', 'SCP', 'HMTEMP', 'LIME', 'DOLO', 'ORE', 'OXY']].values
        x_input_carbon = data[['HMA_C','HMA_MN','HMA_S','HMA_P','HMA_SI','HMA_P','HMA_CR','HMA_TI','BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY']].values
        x_input_phos = data[['HMA_C','HMA_MN','HMA_P','HMA_S','HMA_SI','BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY']].values
        x_input_temp_con = data[['HMA_C', 'HMA_MN', 'HMA_S', 'HMA_P', 'HMA_SI', 'HMA_P', 'HMA_CR', 'HMA_TI','BLOW_DUR', 'HMWT', 'SCP', 'HMTEMP', 'LIME', 'DOLO', 'ORE', 'OXY','BP_OXY_LC_HT_1', 'BP_OXY_LC_HT_7', 'BP_OXY_LC_HT_12','BP_OXY_FLOW_1', 'BP_OXY_FLOW_7', 'BP_OXY_FLOW_12','BP_WST_CO_1', 'BP_WST_CO_7', 'BP_WST_CO_12','BP_WST_CO2_1', 'BP_WST_CO2_7', 'BP_WST_CO2_12',
          'BP_WST_O2_1', 'BP_WST_O2_7', 'BP_WST_O2_12','BP_OXY_PERC_1', 'BP_OXY_PERC_7', 'BP_OXY_PERC_12','BP_WST_FLOW_1', 'BP_WST_FLOW_7', 'BP_WST_FLOW_12', 'BP_WST_TEMP_1', 'BP_WST_TEMP_7', 'BP_WST_TEMP_12'
          ]].values
        x_input_carbon_con = data[['HMA_C','HMA_MN','HMA_S','HMA_P','HMA_SI','HMA_P','HMA_CR','HMA_TI','BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY','BP_OXY_LC_HT_1','BP_OXY_LC_HT_12',
        'BP_OXY_FLOW_1', 'BP_OXY_FLOW_12','BP_OXY_PERC_1','BP_OXY_PERC_12','BP_WST_FLOW_1','BP_WST_FLOW_12' ]].values
        x_input_phos_con = data[['HMA_C','HMA_MN','HMA_S','HMA_P','HMA_SI','HMA_P','HMA_CR','HMA_TI','BLOW_DUR','HMWT','SCP','HMTEMP','LIME','DOLO','ORE','OXY','BP_OXY_LC_HT_1','BP_OXY_LC_HT_12',
        'BP_OXY_FLOW_1', 'BP_OXY_FLOW_12','BP_OXY_PERC_1','BP_OXY_PERC_12','BP_WST_FLOW_1','BP_WST_FLOW_12' ]].values
    button = tkinter.Button(newWindow1, text="Upload Here", command = open_file, fg="black", height="2", width="10")
    button.pack()
    '''options = tkinter.Label(newWindow1, text="Select Options")
    options.config(font=('Arial', 15))
    options.pack()'''

    OPTIONS = [
        "Endpoint Temperature",
        "Endpoint Carbon",
        "Endpoint Phosphorous",
        "Hybrid Model",
        "Slag Chemistries"
    ]  # etc

    variable = tkinter.StringVar(newWindow1)
    variable.set(OPTIONS[0])  # default value

    w = tkinter.OptionMenu(newWindow1, variable, *OPTIONS)
    w.pack(expand=True)
    w.focus()
    def write_to_csv():
        if variable.get() == OPTIONS[0]:
            # Import the standardization scaler
            scaler = load(open('Temperature ANN Discrete.pkl', 'rb'))
            scaler_con = load(open('Temperature ANN Continuous.pkl','rb'))
            # Transform the input variables by using the scaler
            x = scaler.transform(x_input_temp)
            x_con = scaler_con.transform(x_input_temp_con)
            # Import machine learning model
            model_A = load_model('Temperature ANN Discrete.h5')
            model_B = load_model('Temperature ANN Continuous.h5')
            # Use machine learning model to predict endpoint temperature
            predT = model_A.predict(x)
            result = predT
            predT_con = model_B.predict(x_con)
            result_con = predT_con
            result_theo = theo_t(data)

            with open(file_path) as read_obj, \
                    open('Temperature Model.csv', 'w', newline='') as write_obj:
                all_rows = []
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    all_rows.append(row)
                csv_writer= writer(write_obj)
                all_rows[0].append('Predicted Temperature (Discrete Model)')
                all_rows[0].append('Predicted Temperature (Continuous Model)')
                all_rows[0].append('Predicted Temperature (Theoretical Model)')
                for i in range(len(all_rows)-1):
                    all_rows[i+1].append(result[i][0])
                    all_rows[i + 1].append(result_con[i][0])
                    all_rows[i + 1].append(result_theo[i][0])
                    csv_writer.writerow(all_rows[i])

        if variable.get() == OPTIONS[1]:
            # Import the standardization scaler
            scaler = load(open('Carbon ANN Discrete.pkl', 'rb'))
            scaler_con = load(open('Carbon ANN Continuous.pkl', 'rb'))
            # Transform the input variables by using the scaler
            x = scaler.transform(x_input_carbon)
            x_con = scaler_con.transform(x_input_carbon_con)
            # Import machine learning model
            model_A = load_model('Carbon ANN Discrete.h5')
            model_B = load_model('Carbon ANN Continuous.h5')
            # Use machine learning model to predict endpoint temperature
            predT = model_A.predict(x)
            result = predT
            predT_con = model_B.predict(x_con)
            result_con = predT_con
            result_theo = theo_c(data)

            with open(file_path) as read_obj, \
                    open('Carbon Model.csv', 'w', newline='') as write_obj:
                all_rows = []
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    all_rows.append(row)
                csv_writer = writer(write_obj)
                all_rows[0].append('Predicted Carbon (Discrete Model)')
                all_rows[0].append('Predicted Carbon (Continuous Model)')
                all_rows[0].append('Predicted Carbon (Theoretical Model)')
                for i in range(len(all_rows) - 1):
                    all_rows[i + 1].append(result[i][0])
                    all_rows[i + 1].append(result_con[i][0])
                    all_rows[i + 1].append(result_theo[i])
                    csv_writer.writerow(all_rows[i])

        if variable.get() == OPTIONS[2]:
            # Import the standardization scaler
            scaler = load(open('Phos ANN Discrete.pkl', 'rb'))
            scaler_con = load(open('Phos ANN Continuous.pkl', 'rb'))
            # Transform the input variables by using the scaler
            x = scaler.transform(x_input_phos)
            x_con = scaler_con.transform(x_input_phos_con)
            # Import machine learning model
            model_A = load_model('Phos ANN Discrete.h5')
            model_B = load_model('Phos ANN Continuous.h5')
            # Use machine learning model to predict endpoint temperature
            predT = model_A.predict(x)
            result = predT
            predT_con = model_B.predict(x_con)
            result_con = predT_con
            result_theo = theo_p(data)
            with open(file_path) as read_obj, \
                    open('Phosphorus Model.csv', 'w', newline='') as write_obj:
                all_rows = []
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    all_rows.append(row)
                csv_writer = writer(write_obj)
                all_rows[0].append('Predicted Phosphorus (Discrete Model)')
                all_rows[0].append('Predicted Phosphorus (Continuous Model)')
                all_rows[0].append('Predicted Phosphorus (Theoretical Model)')
                for i in range(len(all_rows) - 1):
                    all_rows[i + 1].append(result[i][0])
                    all_rows[i + 1].append(result_con[i][0])
                    all_rows[i + 1].append(result_theo[i])
                    csv_writer.writerow(all_rows[i])

        if variable.get() == OPTIONS[3]:
            sl_feo, sl_fe, cao_ratio, sio2_ratio, p2o5_ratio, mgo, sl_weight = slag_model(data)
            scaler_c = load(open('Carbon ANN Hybrid.pkl', 'rb'))
            scaler_p = load(open('Phos ANN Hybrid.pkl', 'rb'))
            scaler_t = load(open('Temperature ANN Discrete.pkl', 'rb'))

            # Transform the input variables by using the scaler
            x_t = scaler_t.transform(x_input_temp)

            # Import machine learning model
            model_t = load_model('Temperature ANN Discrete.h5')
            model_p = load_model('Phos ANN Hybrid.h5')
            model_c = load_model('Carbon ANN Hybrid.h5')
            # Use machine learning model to predict endpoint temperature
            predT_ann = model_t.predict(x_t)
            predC_ann = hybrid_ann_c(data,predT_ann,scaler_c,model_c)
            predP_ann = hybrid_ann_p(data,predT_ann,scaler_p,model_p)
            theo_hybrid_t = hybrid_theo_t(sl_fe, cao_ratio, sio2_ratio, p2o5_ratio, mgo,sl_weight,data)
            theo_hybrid_p = hybrid_theo_p (sl_fe, cao_ratio,  p2o5_ratio, predT_ann)
            theo_hybrid_c = hybrid_theo_c(sl_fe, cao_ratio, sio2_ratio, p2o5_ratio, mgo,sl_weight,data)
            with open(file_path) as read_obj, \
                    open('Hybrid Model.csv', 'w', newline='') as write_obj:
                all_rows = []
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    all_rows.append(row)
                csv_writer = writer(write_obj)
                all_rows[0].append('Predicted Temperature (Hybrid ANN Model)')
                all_rows[0].append('Predicted Temperature (Hybrid Theoretical Model)')
                all_rows[0].append('Predicted Carbon (Hybrid ANN Model)')
                all_rows[0].append('Predicted Carbon (Hybrid Theoretical Model)')
                all_rows[0].append('Predicted Phosphorus (Hybrid ANN Model)')
                all_rows[0].append('Predicted Phosphorus (Hybrid Theoretical Model)')
                for i in range(len(all_rows) - 1):
                    all_rows[i + 1].append(predT_ann[i][0])
                    all_rows[i + 1].append(theo_hybrid_t[i][0])
                    all_rows[i + 1].append(predC_ann[i][0])
                    all_rows[i + 1].append(theo_hybrid_c[i])
                    all_rows[i + 1].append(predP_ann[i][0])
                    all_rows[i + 1].append(theo_hybrid_p[i][0])
                    csv_writer.writerow(all_rows[i])

        if variable.get() == OPTIONS[4]:
            sl_feo, sl_fe, cao_ratio, sio2_ratio, p2o5_ratio, mgo ,sl_weight= slag_model(data)
            with open(file_path) as read_obj, \
                    open('Slag Chemistry Model.csv', 'w', newline='') as write_obj:
                all_rows = []
                csv_reader = reader(read_obj)
                for row in csv_reader:
                    all_rows.append(row)
                csv_writer = writer(write_obj)
                all_rows[0].append('Predicted SL_FEO')
                all_rows[0].append('Predicted SL_CAO')
                all_rows[0].append('Predicted SL_SIO2')
                all_rows[0].append('Predicted SL_P2O5)')
                all_rows[0].append('Predicted SL_MGO')
                all_rows[0].append('Predicted Slag Weight')
                for i in range(len(all_rows) - 1):
                    all_rows[i + 1].append(sl_feo[i])
                    all_rows[i + 1].append(sio2_ratio[i])
                    all_rows[i + 1].append(p2o5_ratio[i])
                    all_rows[i + 1].append(mgo[i])
                    all_rows[i + 1].append(sl_weight[i])
                    csv_writer.writerow(all_rows[i])
    submit = tkinter.Button(newWindow1, text="Export as CSV", command=write_to_csv, fg="black", height="2", width="10")
    submit.pack()
    submit.focus()

#Manual Input Window
def openNewWindow():

    # Toplevel object which will
    # be treated as a new window
    newWindow = tkinter.Toplevel(window)
    newWindow.geometry("1000x600")
    newWindow.title("Input Parameters")
    chemistries = tkinter.Label(newWindow, text="Hot Metal Chemistry")
    chemistries.config(font=('Roboto', 18))
    chemistries.place(relx=0.076, rely=0.03, anchor='w')
    # sets the geometry of toplevel
    HMA_C = tkinter.Label(newWindow, text="HMA_C (wt%):", width=10)
    HMA_C.place(relx=0.04, rely=0.1, anchor='w')
    HMA_C = tkinter.Entry(newWindow)
    HMA_C.place(relx=0.15, rely=0.1, anchor='w')
    #HMA_C.focus()

    HMA_MN = tkinter.Label(newWindow, text="HMA_MN (wt%):")
    HMA_MN.place(relx=0.04, rely=0.2, anchor='w')
    HMA_MN = tkinter.Entry(newWindow)
    HMA_MN.place(relx=0.15, rely=0.2, anchor='w')
    #HMA_MN.pack(fill='x', expand=True)
    #HMA_MN.focus()

    HMA_P = tkinter.Label(newWindow, text="HMA_P (wt%):")
    HMA_P.place(relx=0.04, rely=0.3, anchor='w')
    HMA_P = tkinter.Entry(newWindow)
    HMA_P.place(relx=0.15, rely=0.3, anchor='w')
    #HMA_P.focus()

    HMA_S= tkinter.Label(newWindow, text="HMA_S (wt%):")
    HMA_S.place(relx=0.04, rely=0.4, anchor='w')
    HMA_S = tkinter.Entry(newWindow)
    HMA_S.place(relx=0.15, rely=0.4, anchor='w')
    #HMA_S.focus()

    HMA_SI = tkinter.Label(newWindow, text="HMA_SI (wt%):")
    HMA_SI.place(relx=0.04, rely=0.5, anchor='w')
    HMA_SI = tkinter.Entry(newWindow)
    HMA_SI.place(relx=0.15, rely=0.5, anchor='w')
    #HMA_SI.focus()

    HMA_CR = tkinter.Label(newWindow, text="HMA_CR (wt%):")
    HMA_CR.place(relx=0.04, rely=0.6, anchor='w')
    HMA_CR = tkinter.Entry(newWindow)
    HMA_CR.place(relx=0.15, rely=0.6, anchor='w')
   # HMA_CR.focus()

    HMA_TI= tkinter.Label(newWindow, text="HMA_TI (wt%):")
    HMA_TI.place(relx=0.04, rely=0.7, anchor='w')
    HMA_TI = tkinter.Entry(newWindow)
    HMA_TI.place(relx=0.15, rely=0.7, anchor='w')
    #HMA_TI.focus()

    operations = tkinter.Label(newWindow, text = "Operational Parameters")
    operations.config(font=('Roboto', 18))
    operations.place(relx=0.41, rely=0.03, anchor='w')

    Blow_Duration = tkinter.Label(newWindow, text="Blow Duration (min):", width=15)
    #Blow_Duration.config(font=('lucida 20 bold italic',14))
    Blow_Duration.place(relx=0.352, rely=0.1, anchor='w')
    Blow_Duration = tkinter.Entry(newWindow)
    Blow_Duration.place(relx=0.52, rely=0.1, anchor='w')
    Blow_Duration.focus()

    Oxygen_Injected = tkinter.Label(newWindow, text="Oxygen Injected (Nm\u00B3):")
    #Oxygen_Injected.config(font=('lucida 20 bold italic',14))
    Oxygen_Injected.place(relx=0.35, rely=0.2, anchor='w')
    Oxygen_Injected = tkinter.Entry(newWindow)
    Oxygen_Injected.place(relx=0.52, rely=0.2, anchor='w')
    Oxygen_Injected.focus()

    Hot_MetalT = tkinter.Label(newWindow, text="HM Temperature(°C):")
    Hot_MetalT.place(relx=0.35, rely=0.3, anchor='w')
    Hot_MetalT = tkinter.Entry(newWindow)
    Hot_MetalT.place(relx=0.52, rely=0.3, anchor='w')
    Hot_MetalT.focus()

    Hot_MetalW = tkinter.Label(newWindow, text="HM Weight (ton):")
    Hot_MetalW.place(relx=0.35, rely=0.4, anchor='w')
    Hot_MetalW = tkinter.Entry(newWindow)
    Hot_MetalW.place(relx=0.52, rely=0.4, anchor='w')
    Hot_MetalW.focus()

    flux = tkinter.Label(newWindow, text="Flux Additions")
    flux.config(font=('Roboto', 18))
    flux.place(relx=0.8, rely=0.03, anchor='w')

    Scrap = tkinter.Label(newWindow, text="Scrap:")
    Scrap.place(relx=0.72, rely=0.1, anchor='w')
    Scrap = tkinter.Entry(newWindow)
    Scrap.place(relx=0.8, rely=0.1, anchor='w')
    Scrap.focus()

    Iron = tkinter.Label(newWindow, text="Iron Ore:")
    Iron.place(relx=0.72, rely=0.2, anchor='w')
    Iron = tkinter.Entry(newWindow)
    Iron.place(relx=0.8, rely=0.2, anchor='w')
    Iron.focus()

    Lime = tkinter.Label(newWindow, text="Lime:")
    Lime.place(relx=0.72, rely=0.3, anchor='w')
    Lime = tkinter.Entry(newWindow)
    Lime.place(relx=0.8, rely=0.3, anchor='w')
    Lime.focus()

    Dolomite = tkinter.Label(newWindow, text="Dolomite:")
    Dolomite.place(relx=0.72, rely=0.4, anchor='w')
    Dolomite = tkinter.Entry(newWindow)
    Dolomite.place(relx=0.8, rely=0.4, anchor='w')
    Dolomite.focus()

    OPTIONS = [
        "Non-Hybrid Model",
        "Hybrid Model"
    ]

    variable = tkinter.StringVar(newWindow)
    variable.set(OPTIONS[0])  # default value

    w = tkinter.OptionMenu(newWindow, variable, *OPTIONS)
    w.place(relx=0.43, rely=0.8, anchor='w')
    w.focus()

    def prediction():
        x_t = np.array([float(HMA_C.get()), float(HMA_MN.get()), float(HMA_P.get()), float(HMA_S.get()),
                        float(HMA_SI.get()), float(HMA_CR.get()), float(HMA_TI.get()), float(Blow_Duration.get()),
                        float(Hot_MetalW.get()), float(Scrap.get()) ,float(Hot_MetalT.get()), float(Lime.get()), float(Dolomite.get()),
                        float(Iron.get()), float(Oxygen_Injected.get())]).reshape(1, -1)

        x_c = np.array([float(HMA_C.get()), float(HMA_MN.get()), float(HMA_S.get()), float(HMA_P.get()),
                        float(HMA_SI.get()),float(HMA_P.get()),float(HMA_CR.get()),float(HMA_TI.get()),float(Blow_Duration.get()),
                        float(Hot_MetalW.get()), float(Scrap.get()), float(Hot_MetalT.get()), float(Lime.get()), float(Dolomite.get()),
                        float(Iron.get()),  float(Oxygen_Injected.get())]).reshape(1, -1)

        x_p = np.array([float(HMA_C.get()), float(HMA_MN.get()), float(HMA_P.get()), float(HMA_S.get()),float(HMA_SI.get()),
                        float(Blow_Duration.get()),float(Hot_MetalW.get()),float(Scrap.get()), float(Hot_MetalT.get()), float(Lime.get()), float(Dolomite.get()),
                        float(Iron.get()),  float(Oxygen_Injected.get())]).reshape(1, -1)

        x_p_2 = np.array([float(HMA_C.get()), float(HMA_MN.get()), float(HMA_S.get()), float(HMA_P.get()),float(HMA_SI.get()),float(HMA_P.get()),float(HMA_CR.get()),
                        float(HMA_TI.get()),float(Blow_Duration.get()),float(Hot_MetalW.get()),float(Scrap.get()), float(Hot_MetalT.get()), float(Lime.get()), float(Dolomite.get()),
                        float(Iron.get()),  float(Oxygen_Injected.get())]).reshape(1, -1)

        if variable.get() == OPTIONS[0]:
            # Import the standardization scaler
            scaler_1 = load(open('Temperature ANN Discrete.pkl', 'rb'))
            scaler_2 = load(open('Carbon ANN Discrete.pkl', 'rb'))
            scaler_3 = load(open('Phos ANN Discrete.pkl', 'rb'))

            # Transform the input variables by using the scaler
            x1 = scaler_1.transform(x_t)
            x2 = scaler_2.transform(x_c)
            x3 = scaler_3.transform(x_p)

            # Import machine learning model
            model_1 = load_model('Temperature ANN Discrete.h5')
            model_2 = load_model('Carbon ANN Discrete.h5')
            model_3 = load_model('Phos ANN Discrete.h5')

            # Use machine learning model to predict endpoint temperature
            pred1 = model_1.predict(x1)
            pred2 = model_2.predict(x2)
            pred3 = model_3.predict(x3)

            result = [pred1,pred2,pred3]

            newWindow1 = tkinter.Toplevel(window)
            newWindow1.geometry("900x600")
            newWindow1.title("Endpoint Prediction (Non Hybrid Model)")

            printresult = tkinter.Label(newWindow1, text="Endpoint Temperature: "+str(result[0][0][0]))
            printresult.place(relx=0.5, rely=0.45, anchor='center')
            printresult.focus()

            printresult = tkinter.Label(newWindow1, text="Endpoint Carbon: " + str(result[1][0][0]))
            printresult.place(relx=0.5, rely=0.5, anchor='center')
            printresult.focus()

            printresult = tkinter.Label(newWindow1, text="Endpoint Phosphorus: " + str(result[2][0][0]))
            printresult.place(relx=0.5, rely=0.55, anchor='center')
            printresult.focus()

        if variable.get() == OPTIONS[1]:
            scaler_c = load(open('Carbon ANN Hybrid.pkl', 'rb'))
            scaler_p = load(open('Phos ANN Hybrid.pkl', 'rb'))
            scaler_t = load(open('Temperature ANN Discrete.pkl', 'rb'))

            # Transform the input variables by using the scaler
            x_t = scaler_t.transform(x_t)
            
            # Import machine learning model
            model_t = load_model('Temperature ANN Discrete.h5')
            model_p = load_model('Phos ANN Hybrid.h5')
            model_c = load_model('Carbon ANN Hybrid.h5')
            # Use machine learning model to predict endpoint temperature
            predT_ann = model_t.predict(x_t)

            x_c = np.append(x_c,predT_ann[0][0]).reshape(1,-1)
            x_p = np.append(x_p_2,predT_ann[0][0]).reshape(1,-1)

            predC_ann = model_c.predict(scaler_c.transform(x_c))
            predP_ann = model_p.predict(scaler_p.transform(x_p))


            result = [predT_ann,predC_ann,predP_ann]

            newWindow1 = tkinter.Toplevel(window)
            newWindow1.geometry("900x600")
            newWindow1.title("Endpoint Prediction (Non Hybrid Model)")

            printresult = tkinter.Label(newWindow1, text="Endpoint Temperature: "+str(result[0][0][0]))
            printresult.place(relx=0.5, rely=0.45, anchor='center')
            printresult.focus()

            printresult = tkinter.Label(newWindow1, text="Endpoint Carbon: " + str(result[1][0][0]))
            printresult.place(relx=0.5, rely=0.5, anchor='center')
            printresult.focus()

            printresult = tkinter.Label(newWindow1, text="Endpoint Phosphorus: " + str(result[2][0][0]))
            printresult.place(relx=0.5, rely=0.55, anchor='center')
            printresult.focus()

    submit1 = tkinter.Button(newWindow, text="Submit", command=prediction)
    submit1.place(relx=0.5, rely=0.9, anchor='w')
    submit1.focus()

# Let's create the Tkinter window.
window = tkinter.Tk(className='Python Examples - Window Size')
window.geometry("900x600")
window.title("Endpoint Predictor")
v = tkinter.IntVar()

# You will first create a division with the help of Frame class and align them on TOP and BOTTOM with pack() method.
top_frame = tkinter.Frame(window).pack()
bottom_frame = tkinter.Frame(window).pack(side = "bottom")
label = tkinter.Label(text="TATA Steel BOF Endpoints Prediction User Interface", font = ("Arial",20))
label.place(relx=0.5, rely=0.2, anchor='center')

# Once the frames are created then you are all set to add widgets in both the frames.
btn1 = tkinter.Button(top_frame, text = "Upload a File", command = filewindow, fg = "black", height = "2", width = "10")
btn1.pack()
btn1.place(relx=0.3, rely=0.4, anchor='center')
#btn1.place(relx=0.653, rely=0.1, relwidth=0.3, anchor='ne')
btn2 = tkinter.Button(top_frame, text = "Manual Input", command = openNewWindow, fg = "black", height = "2", width = "10")
#btn2.pack()
btn2.place(relx=0.7, rely=0.4, anchor='center')

window.mainloop()
