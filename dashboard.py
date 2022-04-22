import json
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt


@st.cache
def get_data(filename):
    df = pd.read_csv(filename, index_col=0)
    liste_id = df.SK_ID_CURR.tolist()
    return liste_id


@st.cache
def get_dataframe(filename):
    df = pd.read_csv(filename, index_col=0)
    return df



def main():
    st.title("Outil de prédiction de crédit")

    liste_id = get_data('data/liste_id.csv')

    st.sidebar.image('data/logo.PNG')

    input_id = st.sidebar.selectbox("Sélectionnez le numéro de client souhaité", liste_id)

    # probabilité
    predict_btn = st.sidebar.button('Prédire')
    if predict_btn:
        response1 = requests.get('https://opc7-backend-lime.herokuapp.com/predict/' + str(input_id))
        prediction = json.loads(response1.content)

        st.sidebar.write("Le modèle retourne une probabilité de solvabilité de :",str(round(prediction['1']['0'],2)),"%")


        if prediction['1']['0'] > 0.5:
            st.sidebar.success('Crédit accordé(threshold 0.5)')
        else:
            st.sidebar.error('Crédit Refusé (threshold 0.5)')

        if prediction['1']['0'] >= 0.4:
            st.sidebar.success('Crédit accordé(threshold 0.4)')
        else:
            st.sidebar.error('Crédit Refusé (threshold 0.4)')

        # Graphique
        response_2 = requests.get("https://opc7-backend-lime.herokuapp.com/graph/" + str(input_id))
        explanation = json.loads(response_2.content)

        explanation = pd.DataFrame(explanation)
        explanation = explanation.sort_values(by='valeur', ascending=False)

        fig, ax = plt.subplots()
        ax.barh(explanation['ticks'], explanation['valeur'], color=['red' if coef < 0 else 'green' for coef in explanation['valeur']])
        ax.grid(visible=True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Comparaison

        liste_col = explanation['ticks'].tolist()

        response_3 = requests.get("https://opc7-backend-lime.herokuapp.com/stats/"+ str(input_id))
        stats_client = json.loads(response_3.content)

        stats_client = pd.DataFrame(stats_client)
        stats_client = stats_client[liste_col]
        stats_client = stats_client.T


        # data sur les groupes
        stat_categ = get_dataframe('data/stats.csv')
        stat_categ = stat_categ.T
        stat_categ = stat_categ[liste_col]
        stat_categ = stat_categ.T

        stat_final = pd.concat([stat_categ, stats_client], axis=1)
        new_colonnes = ['moy gpe 1', 'moy gpe 0', 'client']
        stat_final.set_axis(new_colonnes, axis=1, inplace=True)


        with st.expander("Comparaison par rapport aux groupes ayant et n'ayant pas eu de crédit"):

            col1, col2 = st.columns(2)

            with col1:
                for i in [0,1,2,3,4]:
                    fig, ax = plt.subplots(figsize=(4,2))
                    one_row = stat_final.iloc[i, :]
                    ax.barh([1, 2, 3], one_row.values, color=['lightseagreen', 'mediumturquoise', 'teal'])
                    plt.yticks([1, 2, 3], one_row.index)
                    plt.title(one_row.name)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

            with col2:
                for i in [5,6,7,8,9]:
                    fig, ax = plt.subplots(figsize=(4,2))
                    one_row = stat_final.iloc[i, :]
                    ax.barh([1, 2, 3], one_row.values, color=['lightseagreen', 'mediumturquoise', 'teal'])
                    plt.yticks([1, 2, 3], one_row.index)
                    plt.title(one_row.name)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)





if __name__ == '__main__':
    main()
