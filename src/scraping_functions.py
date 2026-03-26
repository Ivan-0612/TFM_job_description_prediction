from playwright.sync_api import sync_playwright
import pandas as pd
import time
import random
import sys

MAX_PAGINAS = 20  # limite de seguridad para evitar que glassdoor me bloquee la ip por hacer demasiadas peticiones.
# URL base: página de empleos en España, a partir de esta URL aplicaré los filtros de ciudad o puesto para cada búsqueda
BASE_URL = "https://www.glassdoor.es/Empleo/espa%C3%B1a-empleos-SRCH_IL.0,6_IN219.htm"


def cerrar_modal_login(page):
    """
    Esta función se encarga de cerrar cualquier ventana de iniciar sesión o registro que aparezca.
    - El parámetro page es el objeto de Playwright que representa la página web actual.
    - La lista de selectores_cerrar es un conjunto de elementos HTML en las que podría aparecer el botón de cerrar en diferentes versiones o idiomas de la página.
    - El bucle recorre cada selector, lo busca en la página con page.locator(selector).first, si lo encuentra hace clic, hace una pausa para que se cierre y devuelve True, sino reinicia con el siguiente selector
    - Si no encuentra ningun selector, se simula pulsar la tecla Escape para intentar cerrar la ventana, page.keyboard.press("Escape")
    """
    selectores_cerrar = [
        'button[aria-label="Close"]',
        "button.CloseButton",
        '[data-test="close-button"]',
        "button.modal_closeIcon",
        'button[class*="close" i]',
        'button[class*="Close" i]',
        'button[aria-label="Cerrar"]',
        '#LoginModal button[type="button"]',
    ]

    for selector in selectores_cerrar:
        try:
            boton = page.locator(selector).first
            if boton.is_visible(timeout=500):
                boton.click()
                page.wait_for_timeout(1000)
                return True
        except:
            continue
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(1000)
        return True
    except:
        pass
    return False


def aceptar_cookies(page):
    """
    Funcionamiento similar a cerrar_modal_login pero con los selectores de los botones de aceptar cookies. Para cerrar ventanas de aceptar cookies.
    """
    selectores_cookies = [
        "button#onetrust-accept-btn-handler",
        'button[aria-label*="Aceptar" i]',
        'button:has-text("Aceptar las cookies")',
        'button:has-text("Accept")',
        'button:has-text("Aceptar")',
        'button:has-text("Allow all cookies")',
        'button:has-text("Allow")',
    ]

    for selector in selectores_cookies:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=2000):
                btn.click()
                page.wait_for_timeout(1000)
                return True
        except:
            continue
    return False


def encontrar_input(page, selectores):
    """
    Función de apoyo para encontrar el primer campo de input encontrado entre varias opciones de selectores.
    """
    for selector in selectores:
        try:
            campo = page.locator(selector).first
            if campo.is_visible(timeout=1500):
                return campo
        except:
            continue
    return None


def rellenar_campo_location(page, valor):
    """
    Función utilizada para encontrar el campo de ubicacion, limpiarlo y escribir el nuevo valor.
    - Primero busca el campo de ubicación utilizando una lista de posibles selectores html (para adaptarse a diferentes versiones o idiomas de la página).
    - Si encuentra el campo, hace clic para activarlo, hace una pausa, luego selecciona todo el texto existente (Control+a) y lo borra (Backspace), con .press
    - Después escribe el nuevo valor carácter por carácter dejando 80 milisegundos entre cada letra para simular una escritura humana y activar el sistema de autocompletado de la página.
    - Finalmente, intenta seleccionar la primera sugerencia del dropdown que aparece al escribir la ubicación (esperando 3 segundos)
    """
    location_selectors = [
        'input[data-test="search-bar-location-input"]',
        "input#searchBar-location",
        'input[id*="location" i]',
        'input[aria-label*="ubicación" i]',
        'input[aria-label*="location" i]',
        'input[placeholder*="ubicación" i]',
        'input[placeholder*="ciudad" i]',
        'input[placeholder*="location" i]',
        'input[name*="locKeyword" i]',
    ]
    campo = encontrar_input(page, location_selectors)
    if not campo:
        print(f"  ⚠ No se encontró el campo de ubicación")
        return False

    campo.click()
    page.wait_for_timeout(300)
    # Seleccionar todo y borrar
    campo.press("Control+a")
    page.wait_for_timeout(100)
    campo.press("Backspace")
    page.wait_for_timeout(300)
    # Escribir el nuevo valor carácter a carácter para activar autocomplete
    campo.type(valor, delay=80)
    page.wait_for_timeout(2000)

    # Seleccionar primera sugerencia del dropdown
    try:
        sugerencia = page.locator('li[role="option"]').first
        if sugerencia.is_visible(timeout=3000):
            sugerencia.click()
            page.wait_for_timeout(500)
            return True
    except:
        pass  # sino no consigue no da fallo
    # Si no hay dropdown, devolver True igualmente (el texto ya está escrito)
    return True


def rellenar_campo_keyword(page, valor):
    """
    Mismo funcionamiento que rellenar_campo_location pero para el campo de puesto.
    En este caso no hay autocompletado para introducir el puesto exacto que se busca
    """
    keyword_selectors = [
        'input[data-test="search-bar-keyword-input"]',
        "input#searchBar-jobTitle",
        'input[id*="keyword" i]',
        'input[aria-label*="puesto" i]',
        'input[aria-label*="job" i]',
        'input[aria-label*="Búsqueda" i]',
        'input[placeholder*="puesto" i]',
        'input[placeholder*="empleo" i]',
        'input[placeholder*="empresa" i]',
        'input[placeholder*="job" i]',
        'input[name*="keyword" i]',
        'input[name*="sc.keyword" i]',
    ]
    campo = encontrar_input(page, keyword_selectors)
    if not campo:
        print(f"  ⚠ No se encontró el campo de keyword/puesto")
        return False

    campo.click()
    page.wait_for_timeout(300)
    campo.press("Control+a")
    page.wait_for_timeout(100)
    campo.press("Backspace")
    page.wait_for_timeout(300)
    campo.type(valor, delay=60)
    page.wait_for_timeout(500)
    return True


def pulsar_buscar(page):
    """
    Funcion encargada de pulsar el botón de buscar después de rellenar los campos de ubicación o keyword.
    - Primero intenta encontrar el botón de buscar utilizando una lista de posibles selectores
    - Si encuentra el botón, hace clic y espera 4 segundos a que carguen los resultados
    - Si no encuentra ningún botón, intenta pulsar la tecla Enter para activar la búsqueda
    """
    search_selectors = [
        'button[data-test="search-bar-submit"]',
        'button[type="submit"]',
        'button[aria-label*="Buscar" i]',
        'button[aria-label*="Search" i]',
    ]
    for selector in search_selectors:
        try:
            boton = page.locator(selector).first
            if boton.is_visible(timeout=1500):
                boton.click()
                page.wait_for_timeout(4000)
                return True
        except:
            continue
    # Fallback: Enter
    try:
        page.keyboard.press("Enter")
        page.wait_for_timeout(4000)
        return True
    except:
        return False


def extraer_texto(locator, timeout=500):
    """
    Función para extraer texto, con manejo de excepciones para evitar que falle si no encuentra el elemento o no es visible.
    - Recibe un locator de Playwright (donde debería estar el texto)
    - Si hay al menos un elemento y es visible, extrae el texto y lo limpia con strip() para eliminar espacios al inicio y al final
    """
    try:
        if locator.count() > 0 and locator.first.is_visible(timeout=timeout):
            return locator.first.inner_text().strip()
    except:
        pass
    return "N/A"


def extraer_detalle_panel(page):
    """
    Cada vez que se hace clic en una tarjeta de oferta, se abre un panel lateral con información detallada de la oferta.
    Esta función se encarga de extraer la información de este panel de valoración de la empresa y descripcion
    - Primero intenta extraer la valoración de la empresa, buscando el elemento correspondiente y extrayendo su texto
    - Luego intenta pulsar el botón de "mostrar más" para expandir la descripción completa
    - Finalmente, intenta extraer la descripción completa de la oferta, buscando el elemento correspondiente y extrayendo su texto
    No se extrae aqui el salario, fecha o ubicación ya que esa información se obtiene sin abrir el panel lateral, directamente de la tarjeta
    """
    valoracion = "N/A"
    descripcion = "N/A"
    try:
        rating_loc = page.locator(".rating-single-star_RatingText__5fdjN")
        if rating_loc.count() > 0:
            valoracion = rating_loc.first.inner_text().strip()
    except:
        pass
    try:
        mostrar_mas_btn = page.locator('button[data-test="show-more-cta"]')
        if mostrar_mas_btn.count() > 0 and mostrar_mas_btn.is_visible(timeout=1000):
            mostrar_mas_btn.click()
            page.wait_for_timeout(500)
    except:
        pass
    try:
        desc = page.locator(".JobDetails_jobDescription__uW_fK")
        if desc.count() > 0 and desc.is_visible(timeout=2000):
            descripcion = desc.inner_text().strip()
    except:
        pass
    return valoracion, descripcion


def pulsar_mostrar_mas_empleos(page):
    """
    Función para pulsar el botón de "mostrar más empleos" al final de la página de resultados, para cargar más ofertas.
    - Primero define una lista de posibles selectores para el botón de mostrar más
    - Luego recorre cada selector, intentando encontrar el botón correspondiente. Si lo encuentra y es visible, hace clic
    - Espera entre 3 y 5 segundos a que carguen las nuevas ofertas
    - Utiliza la función de cerrar_modal_login para cerrar cualquier ventana de iniciar sesión que pueda aparecer al intentar cargar más ofertas
    - Si no encuentra ningún botón de mostrar más, devuelve False para indicar que no se han podido cargar más ofertas
    """
    selectores = [
        'button[data-test="load-more"]',
        'button:has-text("Mostrar más empleos")',
        'button:has-text("Show more jobs")',
        'button:has-text("Ver más empleos")',
        'button:has-text("Load more jobs")',
        '[data-test="pagination-link-next"]',
        "button.JobsList_buttonWrapper__ticwb",
    ]
    for selector in selectores:
        try:
            boton = page.locator(selector).first
            if boton.is_visible(timeout=2000):
                boton.scroll_into_view_if_needed()
                boton.click()
                page.wait_for_timeout(random.randint(3000, 5000))
                cerrar_modal_login(page)
                return True
        except:
            continue
    return False


def scrapear_glassdoor(
    keyword, location, etiqueta_ciudad, etiqueta_puesto, ruta_archivo=None
):
    """
    Función principal que realiza el proceso de scraping en Glassdoor.

    Primera parte: preparar navegador
    - Se lanza el navegador con Playwright en modo no headless para evitar bloqueos por parte de la página, esto abre el navegador
    - Se crea un contexto, que es básicamente una sesión de navegación temporal y aislada que permiten configurar opciones específicas
        - user_agent se establece para simular un navegador real y evitar bloqueos
        - viewport simula una pantalla de tamaño estándar
        - Se abre una nueva pestaña en blanco con esta configuración
    - Se define el tipo de búsqueda (por ciudad o por puesto) para mostrar mensajes informativos durante el proceso
    - page.goto lleva al navegador a la página de glassdoor especificada
    - se utilizan las funciones de aceptar cookies y cerrar ventana de login para limpiar la página antes de empezar

    Segunda parte: interaccióm
    BLOQUE 1:
    - Se verifica si se ha introducido la localización o el puesto y se llama a la función de rellenar el campo correspondiente
    - Si no se ha podido rellenar el campo, se muestra un mensaje de error y se cierra el navegador
    BLOQUE 2:
    - Utiliza la función de pulsar_buscar para hacer clic en el botón de buscar después de rellenar los campos y cierra posible ventana de login
    - Espera a que aparezcan los resultados, si no aparecen en 15 segundos, muestra un mensaje de error y cierra el navegador
    - Hace una pausa de entre 2 y 4 segundos para simular un comportamiento humano y cierra posible ventana de login
    BLOQUE 3:
    - Empieza el bucle de paginación, que se repetirá hasta un máximo de MAX_PAGINAS para evitar bloqueos por parte de la página
    - Guarda en una lista todas las tarjetas de oferta que aparecen en la página
    - El bucle recorre cada tarjeta:
        - Intenta extrar el título y la empresa de la oferta y limpia el texto
        - Si el título y la empresa ya se ha visto antes (se guarda en un conjunto titulos_vistos para evitar duplicados), se salta la oferta
        - Si es una oferta nueva, mueve la pantalla para hacer visible la tarjeta, hace clic para abrir el panel lateral y una pausa aleatoria (y cierra posible ventana de login)
        - Aqui el proceso se divide, primero extraer la informacion que esta en la tarjeta (localización, salario, fecha) sin abrir el panel lateral, con la función extraer_texto que maneja excepciones para evitar fallos si no encuentra el elemento
        - Luego extrae la información que solo aparece en el panel lateral (valoración y descripcion) con la función extraer_detalle_panel
        - Guarda toda la información extraída en un diccionario
        - Si alguna oferta falla, se captura la excepción y se continúa con la siguiente oferta sin interrumpir el proceso
        - Se va guardando cada pagina en el csv por si surgen errores no perder toda la información
        - Si no se encontraron ofertas nuevas y el boton mostrar más empleos no funciona, terminal el bucle de paginación y cierra el navegador
    """
    datos_extraidos = []  # lista vacia donde se iran guardando las ofertas extraidas
    titulos_vistos = (
        set()
    )  # un conjunto o set que permite no guardar la misma oferta dos veces

    # -----------------Primera parte-------------------

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        tipo_busqueda = f"ciudad={location}" if location else f"puesto={keyword}"

        try:
            page.goto(BASE_URL, timeout=80000)
        except:
            print(f"  Timeout al cargar Glassdoor")
            browser.close()
            return datos_extraidos

        page.wait_for_timeout(random.randint(1000, 2000))
        aceptar_cookies(page)
        cerrar_modal_login(page)

        # -----------------Segunda parte-------------------

        # -----------------BLOQUE 1-------------------

        # Editar el campo correspondiente sobre la página ya cargada
        if location:
            ok = rellenar_campo_location(page, location)
        elif keyword:
            ok = rellenar_campo_keyword(page, keyword)
        else:
            ok = False

        if not ok:
            print(f"  No se pudo editar el formulario para {tipo_busqueda}")
            browser.close()
            return datos_extraidos

        # -----------------BLOQUE 2-------------------

        # Pulsar buscar
        pulsar_buscar(page)
        cerrar_modal_login(page)

        # Esperar a que aparezcan resultados
        try:
            page.wait_for_selector('li[data-test="jobListing"]', timeout=15000)
        except:
            print(f"  No se encontraron ofertas para {tipo_busqueda}")
            browser.close()
            return datos_extraidos

        time.sleep(random.uniform(2, 4))
        cerrar_modal_login(page)

        # -----------------BLOQUE 3-------------------

        pagina_actual = 0
        while pagina_actual < MAX_PAGINAS:
            pagina_actual += 1
            tarjetas = page.locator('li[data-test="jobListing"]').all()

            nuevas = 0
            for index, tarjeta in enumerate(tarjetas):
                try:
                    titulo_check = tarjeta.locator('[data-test="job-title"]')
                    if (
                        titulo_check.count() == 0
                    ):  # sino tiene titulo pasa a la siguiente oferta
                        continue
                    titulo_txt = titulo_check.inner_text().strip()

                    empresa_check = tarjeta.locator(
                        ".EmployerProfile_compactEmployerName__9MGcV"
                    )
                    empresa_txt = (
                        empresa_check.inner_text().strip()
                        if empresa_check.count() > 0
                        else ""
                    )

                    clave = f"{titulo_txt}||{empresa_txt}"
                    if clave in titulos_vistos:
                        continue
                    titulos_vistos.add(clave)
                    nuevas += 1

                    tarjeta.scroll_into_view_if_needed()
                    tarjeta.click()
                    page.wait_for_timeout(random.randint(600, 1200))
                    cerrar_modal_login(page)

                    localizacion = extraer_texto(
                        tarjeta.locator('[data-test="emp-location"]')
                    )
                    salario = extraer_texto(
                        tarjeta.locator('[data-test="detailSalary"]')
                    )
                    if salario == "N/A":
                        salario = "No especificado"
                    fecha = extraer_texto(tarjeta.locator('[data-test="job-age"]'))

                    valoracion, descripcion = extraer_detalle_panel(page)

                    datos_extraidos.append(
                        {
                            "Titulo": titulo_txt,
                            "Empresa": empresa_txt if empresa_txt else "N/A",
                            "Localizacion": localizacion,
                            "Salario": salario,
                            "Fecha_Publicacion": fecha,
                            "Valoracion_Empresa": valoracion,
                            "Descripcion_Completa": descripcion,
                            "Busqueda_Ciudad": etiqueta_ciudad,
                            "Busqueda_Puesto": etiqueta_puesto,
                        }
                    )

                except Exception as e:
                    continue

            if ruta_archivo and len(datos_extraidos) > 0:
                df_temp = pd.DataFrame(datos_extraidos)
                # Limpiamos los saltos de línea del salario
                df_temp["Salario"] = df_temp["Salario"].str.replace(
                    "\n", " ", regex=False
                )
                # Guardamos el CSV sobrescribiendo con los nuevos datos
                df_temp.to_csv(ruta_archivo, index=False, encoding="utf-8-sig")

            if nuevas == 0 or not pulsar_mostrar_mas_empleos(page):
                break

        print(f"  -> Extraidas {len(datos_extraidos)} ofertas para {tipo_busqueda}")
        browser.close()
        return datos_extraidos


if __name__ == "__main__":
    output_path = sys.argv[1]
    etiqueta_ciudad = sys.argv[2]
    etiqueta_puesto = sys.argv[3]
    keyword = sys.argv[4]  # vacío si es búsqueda por ciudad
    location = sys.argv[5]  # vacío si es búsqueda por puesto

    resultados = scrapear_glassdoor(
        keyword, location, etiqueta_ciudad, etiqueta_puesto, output_path
    )
    if resultados:
        df = pd.DataFrame(resultados)
        df["Salario"] = df["Salario"].str.replace("\n", " ", regex=False)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Guardados {len(resultados)} resultados en {output_path}")
    else:
        pd.DataFrame(
            columns=[
                "Titulo",
                "Empresa",
                "Localizacion",
                "Salario",
                "Fecha_Publicacion",
                "Valoracion_Empresa",
                "Descripcion_Completa",
                "Busqueda_Ciudad",
                "Busqueda_Puesto",
            ]
        ).to_csv(output_path, index=False, encoding="utf-8-sig")
        print("Sin resultados.")
