import logging

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Crear un manejador de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Crear un manejador de archivo
    file_handler = logging.FileHandler('app.log')
    file_handler.setLevel(logging.DEBUG)

    # Crear un formateador
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Agregar los manejadores al logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

     # Silenciar mensajes de DEBUG de bibliotecas de terceros
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
