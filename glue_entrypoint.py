import logging
import os
import sys
import glob
import zipfile

# ==========================================================
# 1. CONFIGURACIÓN DE LOGGING (CloudWatch)
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

def main() -> None:
    # ==========================================================
    # 2. SOLUCIÓN AL ERROR 'src' (EXTRACCIÓN MANUAL)
    # ==========================================================
    # Localizamos la carpeta que nos dio éxito en el diagnóstico
    glue_lib_paths = glob.glob('/tmp/glue-python-libs-*/')
    
    if glue_lib_paths:
        root_path = glue_lib_paths[0]
        # Buscamos el archivo que vimos en tu diagnóstico
        zip_file_path = os.path.join(root_path, 'extra_libs.zip')
        
        if os.path.exists(zip_file_path):
            logger.info(f"Detectado archivo: {zip_file_path}. Extrayendo...")
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)
            
            # Agregamos la ruta al sistema para que encuentre 'src'
            sys.path.insert(0, root_path)
            
            # Si el ZIP tenía una carpeta extra llamada 'extra_libs' dentro
            nested_folder = os.path.join(root_path, 'extra_libs')
            if os.path.exists(nested_folder):
                sys.path.insert(0, nested_folder)
            
            logger.info(" Librerías extraídas y sys.path actualizado.")
    else:
        # Fallback para local
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 3. IMPORTS DIFERIDOS (Solo después de extraer el ZIP)
    try:
        from src.database.operations import DatabaseOperations
        from src.scraper.scrapers import MoltbookScraper
        logger.info(" Módulos 'src' cargados con éxito.")
    except ImportError as e:
        logger.error(f" Error fatal: No se encontró 'src'. PATH: {sys.path}")
        sys.exit(1)

    # 4. PARÁMETROS Y ENTORNO
    args = {}
    try:
        from awsglue.utils import getResolvedOptions
        args = getResolvedOptions(
            sys.argv,
            ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"],
        )
    except Exception as e:
        logger.warning(f"getResolvedOptions falló: {e}")

    # Mapeo de variables para el RDS Público
    os.environ["DB_HOST"] = args.get("DB_HOST", "moltbook.ce3qywi6qo2c.us-east-1.rds.amazonaws.com")
    os.environ["DB_NAME"] = args.get("DB_NAME", "postgres") # Base inicial
    os.environ["DB_USER"] = args.get("DB_USER", "postgres")
    os.environ["DB_PASSWORD"] = args.get("DB_PASSWORD", "")

    batch_size = int(args.get("BATCH_SIZE", 5000))
    logger.info(f"Iniciando ETL en host: {os.environ['DB_HOST']}")

    # 5. EJECUCIÓN (DDL + SCRAPING)

    try:
        # 1. Crear tablas en RDS Público
        db_ops = DatabaseOperations(use_postgres=True, chunk_size=batch_size)
        db_ops.ensure_tables()
        logger.info("Tablas verificadas/creadas.")

        # 2. Scraper sin Playwright (requests + BeautifulSoup) para evitar pip/cache en Glue
        with MoltbookScraper(db_ops=db_ops, headless=True, use_playwright=False) as scraper:
            result = scraper.scrape_all(
                max_users=int(args.get("MAX_USERS", 100)),
                max_submolts=50,
                max_posts=500,
                max_comments=1000
            )
        
        logger.info(f" ETL finalizado con éxito: {result}")

    except Exception as e:
        logger.error(f"Fallo en la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()