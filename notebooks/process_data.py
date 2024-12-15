from AntennaDesign.utils import DataPreprocessor
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument('--data_path', type=str, help='Path to the raw data',
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\raw_data_130k_200k')
    parser.add_argument('--destination_path', type=str, help='Path to the destination directory for processed data',
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--process_antenna', action='store_true', help='Process antenna data')
    parser.add_argument('--process_environment', action='store_true', help='Process environment data')
    parser.add_argument('--process_radiation', action='store_true', help='Process radiation data')
    parser.add_argument('--radiation_mode', action='directivity', help='Process directivity data')
    parser.add_argument('--process_gamma', action='store_true', help='Process gamma data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    debug = args.debug
    preprocessor = DataPreprocessor(data_path=args.data_path, destination_path=args.destination_path)

    preprocessor.antenna_preprocessor(debug=debug) if args.process_antenna else None
    preprocessor.environment_preprocessor(debug=debug) if args.process_environment else None
    preprocessor.radiation_preprocessor(debug=debug, mode=args.radiation_mode) if args.process_radiation else None
    preprocessor.gamma_preprocessor(debug=debug) if args.process_gamma else None
