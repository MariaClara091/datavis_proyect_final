import streamlit as st
st.set_page_config(page_title="Visualización Proyecto UVB - Autores: María Clara Ávila y Mateo José Giraldo", layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, shapiro, levene, f_oneway

# === Config ===
DATAFILE = "DATASET_PROY_OFICIAL.csv"  
sns.set(style="whitegrid")

# --- Helpers / Caching ---
@st.cache_data
def load_data(path=DATAFILE):
    df = pd.read_csv(path)
    return df

def safe_corr(x, y):
    # calcula Pearson manejando NaNs y tamaños pequeños
    valid = (~x.isna()) & (~y.isna())
    if valid.sum() < 3:
        return np.nan, np.nan
    try:
        r, p = pearsonr(x[valid], y[valid])
    except Exception:
        return np.nan, np.nan
    return r, p

def fig_from_pairplot(df, cols):
    # pairplot devuelve un objeto seaborn PairGrid con atributo fig
    g = sns.pairplot(df[cols].dropna())
    fig = g.fig
    return fig

# --- Load ---
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a:", [
    "Inicio / Contexto",
    "Carga & ETL",
    "Descripción (Univariado)",
    "Visualizaciones (Originales)",
    "Relaciones (Bivariado)",
    "Pruebas estadísticas",
    "Conclusiones",
    "Datos crudos"
])

# Try load data, show friendly error if fails
try:
    datos = load_data()
except FileNotFoundError:
    st.sidebar.error(f"No se encontró {DATAFILE}. Coloca el CSV en la carpeta de la app.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error cargando datos: {e}")
    st.stop()

# Provide some shortcuts for column names used in notebook
# If these columns don't exist, the app will handle it gracefully where used.
cols_present = set(datos.columns)

# --- PREPROCESSING (keep same names used in notebook) ---
# Drop exact duplicate rows (documented)
duplicados = int(datos.duplicated().sum())

# Create datos_limpios following the notebook logic: remove impossible (<=0) for some vars
datos_limpios = datos.copy()
for col in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"]:
    if col in datos_limpios.columns:
        try:
            datos_limpios = datos_limpios[datos_limpios[col] > 0]
        except Exception:
            # if non-numeric, coerce then filter
            datos_limpios[col] = pd.to_numeric(datos_limpios[col], errors="coerce")
            datos_limpios = datos_limpios[datos_limpios[col] > 0]

# If MONTH column is 'MO' or similar, ensure it's present and categorical
if "MO" in datos_limpios.columns:
    try:
        datos_limpios["MO"] = datos_limpios["MO"].astype(int)
    except Exception:
        pass

# === Pages ===
if page == "Inicio / Contexto":
    st.title("Contexto y diseño del EDA")
    st.markdown("""
    **Problema:** El aumento de la radiación solar UVB puede representar un riesgo para la salud humana y el ambiente.  
    **Objetivo general:** Analizar la variabilidad temporal y las relaciones entre radiación UVB, presión, precipitación y otras variables atmosféricas.
    """)
    st.markdown("**Hipótesis específicas:**\n- H1: La radiación UVB presenta picos durante los meses secos.\n- H2: Existe correlación negativa entre precipitación y radiación UVB.\n- H3: La presión superficial y la radiación UVB están asociadas de forma indirecta.")
    st.markdown("**Diccionario de variables (ejemplo):**")
    st.write(pd.DataFrame([
        ["YEAR","Año de registro","Categórica","-"],
        ["MO","Mes de registro","Categórica","-"],
        ["DY","Día de registro","Categórica","-"],
        ["HR","Hora del registro","Categórica","h"],
        ["ALLSKY_SFC_UVB","Irradiancia UVB en superficie bajo cielo total","Numérica","W/m²"],
        ["ALLSKY_SFC_LW_DWN","Irradiancia de onda larga descendente","Numérica","W/m²"],
        ["PRECTOTCORR","Precipitación total corregida","Numérica","mm/h"],
        ["PS","Presión superficial","Numérica","kPa"],
    ], columns=["Variable","Descripción","Tipo","Unidad"]))
    st.markdown("---")
    st.write("Resumen rápido del dataset:")
    st.write(f"Filas originales: {datos.shape[0]}  —  Columnas: {datos.shape[1]}  —  Duplicados: {duplicados}")
    st.write("Primeras filas:")
    st.dataframe(datos.head(10))

if page == "Carga & ETL":
    st.title("⚙️ ETL (Extract, Transform, Load)")
    st.markdown("""
    Se muestra revisión de duplicados, porcentajes de nulos por variable, y limpieza aplicada (remoción de valores imposibles).
    """)
    st.subheader("Duplicados y nulos")
    st.write(f"Duplicados totales: **{duplicados}**")
    nulos = datos.isna().sum().reset_index()
    nulos.columns = ["Variable", "Nulos"]
    nulos["% Nulos inicial"] = (nulos["Nulos"] / len(datos)) * 100
    nulos["Tipo de ausencia"] = np.where(nulos["Nulos"] > 0, "MCAR/No MCAR", "Sin ausencias")
    nulos["Método imputación"] = np.where(nulos["Nulos"] > 0, "Media/interpolación (ej.)", "No aplica")
    nulos["Métrica validación"] = np.where(nulos["Nulos"] > 0, "RMSE (ej.)", "No aplica")
    st.dataframe(nulos)

    st.subheader("Limpieza aplicada")
    st.write("Se eliminaron filas con valores no-positivos en variables: ALLSKY_SFC_UVB, ALLSKY_SFC_LW_DWN, PRECTOTCORR, PS (si estaban presentes).")
    st.write(f"Filas después de limpieza: **{datos_limpios.shape[0]}**")

    st.markdown("Si deseas aplicar imputación (media/interpolación), agrega el código correspondiente en el notebook o habilita la opción de imputación aquí (no realizada por defecto).")

if page == "Descripción (Univariado)":
    st.title("Análisis descriptivo (univariado)")
    st.markdown("Resumen estadístico (datos limpios):")
    st.dataframe(datos_limpios.describe(include='all').transpose())

    st.markdown("Histogramas y boxplots de variables clave:")
    cols_to_plot = [c for c in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in datos_limpios.columns]
    for c in cols_to_plot:
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        sns.histplot(datos_limpios[c].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f"Histograma {c}")
        sns.boxplot(x=datos_limpios[c].dropna(), ax=ax[1])
        ax[1].set_title(f"Boxplot {c}")
        st.pyplot(fig)

if page == "Visualizaciones (Originales)":
    st.title("Visualizaciones (originales del notebook)")
    st.markdown("Se muestran las gráficas exploratorias originales. Ajusta filtros en el sidebar si quieres.")
    # Example: pairplot (note: may be heavy)
    st.markdown("**Pairplot (variables seleccionadas)**")
    pair_cols = [c for c in ["ALLSKY_SFC_UVB", "ALLSKY_SFC_LW_DWN", "PRECTOTCORR", "PS"] if c in datos_limpios.columns]
    if len(pair_cols) >= 2:
        with st.spinner("Generando pairplot... (puede tardar)"):
            try:
                fig = fig_from_pairplot(datos_limpios, pair_cols)
                st.pyplot(fig)
            except Exception as e:
                st.write("No se pudo generar pairplot:", e)
    else:
        st.write("No hay suficientes columnas para pairplot.")

    st.markdown("**Matriz de correlación**")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(datos_limpios.corr(), annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("**Distribución mensual de UVB (boxplot por mes)**")
    if "MO" in datos_limpios.columns and "ALLSKY_SFC_UVB" in datos_limpios.columns:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.boxplot(x="MO", y="ALLSKY_SFC_UVB", data=datos_limpios, ax=ax)
        ax.set_title("Distribución mensual de radiación UVB")
        st.pyplot(fig)
    else:
        st.write("Faltan columnas MO o ALLSKY_SFC_UVB para este gráfico.")

if page == "Relaciones (Bivariado)":
    st.title("Relaciones bivariadas")
    st.markdown("Selecciona la relación que quieres ver:")

    rel = st.selectbox("Relación", ["PS vs UVB", "PRECTOTCORR vs UVB", "UVB vs Mes (boxplot)"])
    if rel == "PS vs UVB":
        st.subheader("Presión superficial vs Radiación UVB")
        if not (("PS" in datos_limpios.columns) and ("ALLSKY_SFC_UVB" in datos_limpios.columns)):
            st.warning("Faltan las columnas PS o ALLSKY_SFC_UVB en el dataset.")
        else:
            use_lowess = st.checkbox("Usar LOWESS (requiere statsmodels instalado)", value=False)
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=datos_limpios, x="PS", y="ALLSKY_SFC_UVB", alpha=0.4, s=30, edgecolor=None, ax=ax)
            try:
                if use_lowess:
                    # will raise if statsmodels not installed; catch below
                    sns.regplot(data=datos_limpios, x="PS", y="ALLSKY_SFC_UVB", scatter=False, lowess=True, ax=ax, color="blue")
                else:
                    sns.regplot(data=datos_limpios, x="PS", y="ALLSKY_SFC_UVB", scatter=False, ax=ax, color="blue")
            except Exception as e:
                # fallback to linear if lowess unavailable
                sns.regplot(data=datos_limpios, x="PS", y="ALLSKY_SFC_UVB", scatter=False, ax=ax, color="blue")
                st.info("LOWESS no disponible; mostrando regresión lineal. Para LOWESS instala statsmodels en el entorno.")
            r, p = safe_corr(datos_limpios["PS"], datos_limpios["ALLSKY_SFC_UVB"])
            if not np.isnan(r):
                ax.text(0.02, 0.95, f"r = {r:.3f}\np = {p:.3e}", transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.set_title("Relación entre Presión Superficial y Radiación UVB")
            ax.set_xlabel("Presión superficial (kPa)")
            ax.set_ylabel("Radiación UVB (W/m²)")
            st.pyplot(fig)

    elif rel == "PRECTOTCORR vs UVB":
        st.subheader("Precipitación (PRECTOTCORR) vs Radiación UVB")
        if not (("PRECTOTCORR" in datos_limpios.columns) and ("ALLSKY_SFC_UVB" in datos_limpios.columns)):
            st.warning("Faltan las columnas PRECTOTCORR o ALLSKY_SFC_UVB en el dataset.")
        else:
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=datos_limpios, x="PRECTOTCORR", y="ALLSKY_SFC_UVB", alpha=0.35, s=30, ax=ax)
            sns.regplot(data=datos_limpios, x="PRECTOTCORR", y="ALLSKY_SFC_UVB", scatter=False, ax=ax, color="green")
            r, p = safe_corr(datos_limpios["PRECTOTCORR"], datos_limpios["ALLSKY_SFC_UVB"])
            if not np.isnan(r):
                ax.text(0.02, 0.95, f"r = {r:.3f}\np = {p:.3e}", transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            ax.set_title("Relación entre Precipitación y Radiación UVB")
            ax.set_xlabel("Precipitación (mm/h)")
            ax.set_ylabel("Radiación UVB (W/m²)")
            st.pyplot(fig)

    elif rel == "UVB vs Mes (boxplot)":
        st.subheader("Distribución mensual de UVB")
        if not (("MO" in datos_limpios.columns) and ("ALLSKY_SFC_UVB" in datos_limpios.columns)):
            st.warning("Faltan las columnas MO o ALLSKY_SFC_UVB en el dataset.")
        else:
            fig, ax = plt.subplots(figsize=(10,4))
            sns.boxplot(x="MO", y="ALLSKY_SFC_UVB", data=datos_limpios, ax=ax)
            ax.set_title("Distribución mensual de radiación UVB")
            ax.set_xlabel("Mes")
            ax.set_ylabel("Radiación UVB (W/m²)")
            st.pyplot(fig)

if page == "Pruebas estadísticas":
    st.title("Pruebas estadísticas")
    st.markdown("Se realizan pruebas de normalidad, homogeneidad y ANOVA (UVB por mes).")

    # Normalidad (Shapiro) para muestra (si es muy grande se muestra en muestra)
    test_cols = [c for c in ["ALLSKY_SFC_UVB", "PRECTOTCORR", "PS"] if c in datos_limpios.columns]
    if test_cols:
        st.subheader("Normalidad (Shapiro-Wilk) — muestra hasta 500 observaciones")
        for c in test_cols:
            serie = datos_limpios[c].dropna()
            muestra = serie.sample(500, random_state=42) if len(serie) > 500 else serie
            try:
                stat, p = shapiro(muestra)
                st.write(f"{c}: W={stat:.3f}, p={p:.3e}")
            except Exception as e:
                st.write(f"{c}: No se pudo aplicar Shapiro (tamaño/presencia NaNs). Error: {e}")

    # Levene & ANOVA UVB por mes
    if ("MO" in datos_limpios.columns) and ("ALLSKY_SFC_UVB" in datos_limpios.columns):
        st.subheader("Homogeneidad (Levene) y ANOVA (UVB por mes)")
        grupos = [g["ALLSKY_SFC_UVB"].dropna().values for _, g in datos_limpios.groupby("MO")]
        try:
            lev_stat, lev_p = levene(*[g for g in grupos if len(g) > 0])
            st.write(f"Levene: stat={lev_stat:.3f}, p={lev_p:.3e}")
        except Exception as e:
            st.write("Levene no pudo calcularse:", e)
        try:
            # ANOVA
            an_stat, an_p = f_oneway(*[g for g in grupos if len(g) > 1])
            st.write(f"ANOVA: stat={an_stat:.3f}, p={an_p:.3e}")
        except Exception as e:
            st.write("ANOVA no pudo calcularse:", e)
    else:
        st.write("Faltan MO o ALLSKY_SFC_UVB para Levene/ANOVA.")

if page == "Conclusiones":
    st.title("Interpretación y conclusiones")
    st.markdown("""
    - Observaciones generales:
      - Revisa si la radiación UVB presenta picos durante meses secos (compara boxplots por mes).
      - Busca correlación negativa entre precipitación y UVB.
      - Revisa relación entre presión superficial y UVB (probablemente dé correlación débil).
    - Limitaciones:
      - Datos de una sola estación; no se incluyeron factores externos (cobertura nubosa, altitud, aerosoles).
      - Si hay valores perdidos se deben documentar y justificar los métodos de imputación.
    """)
    st.markdown("**Próximos pasos recomendados:**\n- Hacer validación cruzada de imputaciones si se imputan valores.\n- Incorporar datos de nubosidad o radiación clara/real para desambiguar efectos.\n- Crear un informe resumido en PDF con los hallazgos principales.")
    st.write("Si deseas, puedo generar el notebook final (.ipynb) o exportar gráficos automáticamente (implementación adicional).")

if page == "Datos crudos":
    st.title("Datos crudos")
    st.write("Vista completa (primeras 200 filas):")
    st.dataframe(datos.head(200))
    st.markdown("Descarga rápida del CSV:")
    with open(DATAFILE, "rb") as f:
        st.download_button("Descargar CSV original", data=f, file_name=DATAFILE)

# Footer / nota
st.markdown("---")
st.caption("App generada a partir del notebook Visualiza_python1.ipynb — adapta rutas/archivos si tu CSV tiene otro nombre. Para activar LOWESS instala statsmodels en el entorno.")
