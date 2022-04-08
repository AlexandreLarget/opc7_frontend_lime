import json
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt


@st.cache
def get_data(filename):
    dataframe = pd.read_csv(filename, index_col=0)
    liste_id = dataframe.SK_ID_CURR.tolist()
    dataframe.drop("TARGET", axis=1)
    return dataframe, liste_id


@st.cache
def get_score_explainer(filename):
    dataframe = pd.read_csv(filename)
    return dataframe


def main():
    st.title("Outil de prédiction de crédit")

    data, liste_id = get_data('data/data_scaled_sample.csv')

    input_id = st.sidebar.selectbox("Sélectionnez le numéro de client souhaité", liste_id)

    id_data = data[data['SK_ID_CURR'] == int(input_id)].drop("SK_ID_CURR", axis=1)
    id_data = id_data.to_dict(orient='records')[0]
    input_dict = {}
    input_dict['id_client'] = input_id

    # probabilité
    predict_btn = st.sidebar.button('Prédire')
    if predict_btn:
        response1 = requests.post('https://opc7-backend.herokuapp.com/predict', data=json.dumps(id_data, allow_nan=True))
        prediction = json.loads(response1.content)

        if prediction['1']['0'] > 0.5:
            st.sidebar.success('Crédit accordé(threshold 0.5)')
        else:
            st.sidebar.error('Crédit Refusé (threshold 0.5')

        if prediction['1']['0'] >= 0.4:
            st.sidebar.success('Crédit accordé(threshold 0.4)')
        else:
            st.sidebar.error('Crédit Refusé (threshold 0.4)')

        # Explication
        response_2 = requests.get("https://opc7-backend.herokuapp.com/gal_exp/" + str(input_id))
        explanation = json.loads(response_2.content)

        explanation = pd.DataFrame(explanation)
        explanation.rename(columns={'0': 'features', '1': 'value'}, inplace=True)
        explanation['value'] = explanation['value'].astype(float)
        explanation['features'] = explanation['features'].str.strip()

        fig, ax = plt.subplots()
        ax.barh(explanation['features'], explanation['value'], color=['red' if coef < 0 else 'green' for coef in explanation['value']])
        ax.grid(visible=True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        score_explainer = get_score_explainer("data/explain.csv")


        exp_detail = explanation.merge(score_explainer, left_on='features', right_on='Row', how='left')
        exp_detail = exp_detail.loc[:,['features', 'value', 'Description']]
        exp_detail = exp_detail.drop_duplicates('features')
        exp_detail = exp_detail.sort_values(by='value', ascending=False, key=abs)
        exp_detail = exp_detail.dropna(axis=0)

        with st.expander("Explications détaillées"):
            for i in range(exp_detail.shape[0]):
                st.write(exp_detail.iloc[i,0])
                st.write("Définition : {}".format(exp_detail.iloc[i, 2]))
                st.write("------------------------------------------------------")
                st.write(" ")


if __name__ == '__main__':
    main()
