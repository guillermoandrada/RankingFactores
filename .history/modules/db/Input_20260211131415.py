import os
import re
import sys
import pandas as pd

from DataBase import save_dataframe_to_sql

# --- CONFIGURACIÓN DE ENTRADA ---

# Nombre exacto que salió en tu comando 'dir'
ARCHIVO_POR_DEFECTO = "2024 Q4.xlsx"

# Columnas que identifican a la empresa (las demás son métricas)
# (Se duplica aquí para validación del Excel antes de llamar al backend)
FIXED_COLUMNS = [
    "Ticker",
    "Long Name",
    "GICS Sector Name",
    "GICS Industry Group Name",
    "Market Cap (USD)",
]


# --- FUNCIONES DE ENTRADA / LECTURA ---


def get_period_from_name(filename: str) -> str:
    """Intenta sacar '2024 Q4' del nombre del archivo."""
    base = os.path.basename(filename)
    # Busca 4 dígitos + Q + número (ej: 2024 Q4)
    match = re.search(r"(\d{4}\s*Q[1-4])", base, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return "DESCONOCIDO"


def get_index_code_from_excel(filepath: str) -> str | None:
    """
    Lee las filas superiores del Excel (sin cabecera) y busca una celda que ponga
    'Universe Name' (o parecido). Devuelve el valor de la celda justo debajo
    en la misma columna, por ejemplo 'B500'.

    Si no se encuentra nada, devuelve None.
    """
    try:
        # Leemos solo las primeras filas, sin interpretar cabeceras
        top = pd.read_excel(filepath, header=None, nrows=10)
    except Exception:
        # Si ni siquiera se puede leer como Excel, no hay código de índice
        return None

    # Buscamos "Universe Name" (insensible a mayúsculas/minúsculas/espacios)
    for row_idx in range(top.shape[0] - 1):  # -1 para poder mirar la fila siguiente
        for col_idx in range(top.shape[1]):
            val = top.iat[row_idx, col_idx]
            if isinstance(val, str) and "universe" in val.lower() and "name" in val.lower():
                # Tomamos la celda de debajo como código de índice
                below = top.iat[row_idx + 1, col_idx]
                if isinstance(below, str) and below.strip():
                    return below.strip()
                # También aceptamos números/códigos no string
                if below is not None and str(below).strip():
                    return str(below).strip()

    return None


def read_data(filepath: str) -> pd.DataFrame:
    """Lee Excel o CSV de forma inteligente para los datos de securities/métricas."""
    print(f"Leyendo archivo (datos de tabla): {filepath} ...")

    # 1. Intentar como Excel
    try:
        # header=3 significa que los títulos están en la fila 4
        return pd.read_excel(filepath, header=3)
    except Exception as e_excel:
        print(f"No se pudo leer como Excel ({e_excel}). Probando como CSV...")

    # 2. Intentar como CSV (Plan B)
    try:
        return pd.read_csv(filepath, header=3)
    except Exception:
        raise ValueError(
            "CRÍTICO: No se pudo leer el archivo ni como Excel ni como CSV."
        )


def process_file(filepath: str) -> None:
    """
    Orquesta la lectura y validación del archivo,
    y delega el guardado en la función del backend.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"El archivo NO está en esta carpeta: {filepath}")

    # 0. Obtener código de índice desde la parte superior del Excel (ej. 'B500')
    index_code = get_index_code_from_excel(filepath)
    print(f"--> Código de índice detectado en Excel: {index_code}")

    # 1. Leer tabla principal de datos
    df = read_data(filepath)

    # 2. Validaciones básicas de columnas
    if "Ticker" not in df.columns:
        raise ValueError(
            f"El archivo no tiene la columna 'Ticker'. Columnas encontradas: {list(df.columns)}"
        )

    missing_cols = [c for c in FIXED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_cols}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # 3. Limpiar filas sin Ticker
    df = df.dropna(subset=["Ticker"])

    # 4. Obtener periodo desde el nombre del archivo
    period = get_period_from_name(filepath)
    print(f"--> Periodo detectado: {period}")

    # 5. Guardar en la base de datos usando el backend (incluye índice si lo hay)
    save_dataframe_to_sql(df, period, index_code=index_code)


# --- PUNTO DE ENTRADA ---


if __name__ == "__main__":
    # Si le pasas un archivo por comando lo usa, si no, usa el de defecto
    archivo = sys.argv[1] if len(sys.argv) > 1 else ARCHIVO_POR_DEFECTO

    try:
        process_file(archivo)
    except Exception as e:
        print("\n--- ERROR ---")
        print(e)
        print("-------------")
        # Pausa para que leas el error si se cierra rápido
        input("Presiona ENTER para salir...")