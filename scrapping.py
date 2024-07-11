import requests
from bs4 import BeautifulSoup
import os
import hashlib
import time





def buscar_imagenes(query, directorio):
    """
    Función que busca imágenes en un buscador y las guarda en un directorio.

    Args:
        query: El término de búsqueda.
        directorio: El directorio donde se guardarán las imágenes.
    """
   
    # Crear el directorio si no existe
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # URL del buscador
    url_base = "https://www.bing.com/images/search"

    # Número de páginas a scrapear
    num_paginas = 30  # Ajustar según se desee

    # Mantener un conjunto de hashes MD5 de imágenes descargadas
    imagenes_descargadas = set()

    for pagina in range(1, num_paginas + 1):
        # Parámetros de la búsqueda
        params = {
            "q": query,
            "first": pagina * 28  # Bing muestra 28 imágenes por página
        }

        # Realizar la petición HTTP
        response = requests.get(url_base, params=params)

        # Comprobar si la petición fue exitosa
        if response.status_code == 200:
            # Parsear el HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Encontrar las imágenes
            imagenes = soup.find_all("img", {"class": "mimg"})

            for imagen in imagenes:
                # Obtener la URL de la imagen
                url_imagen = imagen.get("src")
                if not url_imagen:
                    url_imagen = imagen.get("data-src")

                if url_imagen:
                    # Descargar la imagen y guardarla si no es repetida
                    
                    descargar_imagen(url_imagen, directorio, imagenes_descargadas)
                    
        else:
            print(f"Error al buscar imágenes en la página {pagina}: {response.status_code}")

def descargar_imagen(url_imagen, directorio, imagenes_descargadas, max_reintentos=3):
    """
    Función que descarga una imagen y la guarda en un directorio.

    Args:
        url_imagen: La URL de la imagen.
        directorio: El directorio donde se guardará la imagen.
        imagenes_descargadas: Conjunto de hashes MD5 de imágenes ya descargadas.
        max_reintentos: Número máximo de reintentos en caso de fallo.
    """

    reintento = 0
    counter =0
    while reintento < max_reintentos:
        try:
            # Descargar la imagen con un tiempo de espera
            response = requests.get(url_imagen, timeout=10)

            # Comprobar si la descarga fue exitosa
            if response.status_code == 200:
                # Generar el hash MD5 de la imagen descargada
                hash_md5 = hashlib.md5(response.content).hexdigest()

                # Verificar si la imagen ya se ha descargado
                if hash_md5 not in imagenes_descargadas:
                    # Añadir el hash al conjunto de imágenes descargadas
                    imagenes_descargadas.add(hash_md5)

                    # Generar un nombre de archivo seguro utilizando el hash MD5
                    nombre_imagen = hash_md5 + ".jpg"
                    
                    
                    # Guardar la imagen
                    with open(os.path.join(directorio, nombre_imagen), "wb") as f:
                        f.write(response.content)
                    print(f"Imagen guardada: {nombre_imagen}" )
                    
                else:
                    print(f"Imagen repetida: {url_imagen}")
                return 

            else:
                print(f"Error al descargar imagen: {url_imagen}, Status Code: {response.status_code}")
                return

        except requests.exceptions.RequestException as e:
            print(f"Error al descargar imagen: {url_imagen}, Reintento: {reintento + 1}/{max_reintentos}")
            print(e)
            reintento += 1
            time.sleep(5)  # Esperar 5 segundos antes de reintentar

    print(f"Fallo al descargar la imagen después de {max_reintentos} intentos: {url_imagen}")

# Buscar imágenes de "horse" en el directorio "dataset/horse/"
buscar_imagenes("cascara de platano", "dataset/organic/")
buscar_imagenes("manzana mordida", "dataset/organic/")
buscar_imagenes("fruta", "dataset/organic/")
buscar_imagenes("verdura", "dataset/organic/")


