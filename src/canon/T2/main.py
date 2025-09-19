from canon import config
import argparse
from pathlib import Path
import logging


from canon.utils import image_utils
from canon.T2.process import feature_extraction, epipolar_geometry
from canon.T2.plotting import visualization

MIN_MATCHES = 30  # Mínimo de matches por par


def load_images(args):
    image_dir = Path(args.image_dir)
    
    image_dir_full = image_utils.BASE_DATA_PATH / image_dir
    
    # Verificar se a pasta existe
    if not image_dir_full.exists() or not image_dir_full.is_dir():
        logger.error(f"Pasta de imagens não encontrada: {image_dir}")
        exit(1)
    
    logger.info(f"Pasta de imagens: {image_dir}")
    
    images = image_utils.load_images(image_dir)
    
    logger.info(f"Carregadas {len(images)} imagens")
    
    return images


def extract_features(args, images):
    logger.info(f"Extraindo features usando detector: {args.detector}")
    
    if args.detector == "SIFT":
        features_data = feature_extraction.extract_sift_for_3d(images, save_visualizations=False)
    elif args.detector == "ORB":
        features_data = feature_extraction.extract_orb_for_3d(images, save_visualizations=False)
    elif args.detector == "AKAZE":
        features_data = feature_extraction.extract_akaze_for_3d(images, save_visualizations=False)
        
        
    features = {name: (data['keypoints'], data['descriptors']) 
               for name, data in features_data.items()}
    
    # Exemplo de log de saída
    for img_name, (kp, desc) in features.items():
        logger.debug(f"{img_name}: {len(kp)} keypoints extraídos")

    logger.info("Features extraídas com sucesso")
    
    return features


def match_images(args, features):
    matcher = epipolar_geometry.ImagePairMatcher(
        matcher_type="BF",  # Brute Force
        ratio_threshold=0.75,
        cross_check=False  # Desabilitado para usar ratio test
    )

    logger.info("Matcher configurado: Brute Force com Lowe's ratio test (0.75)")
    
    logger.info("Iniciando match das imagens")

    match_results = epipolar_geometry.match_image_collection(
        features=features,
        matcher=matcher,
        max_pairs=None,
        min_matches=MIN_MATCHES
    )
    
    logger.info(f"\nEmparelhamento concluído!")
    logger.info(f"Pares bem-sucedidos: {len(match_results)}")
    
    return match_results


if __name__ == "__main__":
    # Configurar logger
    logger = config.setup_logger("3D main pipeline")
    
    # Configurar argparse para receber argumentos
    parser = argparse.ArgumentParser(description="Pipeline de reconstrução 3D")
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Caminho para a pasta com as imagens",
        required=True
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["SIFT", "ORB", "AKAZE"],
        default="SIFT",
        help="Tipo de detector de features a ser usado"
    )
    args = parser.parse_args()
    
    # --- Carregar imagens ---
    images = load_images(args)
    
    # --- Extrair features ---
    features = extract_features(args, images)
    
    match_results = match_images(args, features)
    
